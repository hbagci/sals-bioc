import os
import sys
import io
import base64
import random
import tempfile
import traceback
import uuid as _uuid
from collections import Counter

import copy

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify, send_from_directory

# ── Load the algorithm modules from the parent directory ────────────────────
HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import emcm_ps       # noqa: E402
import lstm          # noqa: E402
import lifelong_lstm # noqa: E402

lstm.set_device('cpu')

# ── Flask app (serves static frontend from the same directory) ──────────────
app = Flask(__name__, static_folder=HERE, static_url_path="")

# ── In-memory LSTM model cache keyed by (side, model_id) ───────────────────
_LSTM_MODELS = {}

# ── In-memory lifelong session cache keyed by (side, session_id) ───────────
_LIFELONG_SESSIONS = {}


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


def _load_dataframe(body):
    csv_text = body.get("csv")
    if not csv_text:
        raise ValueError("Missing 'csv' in request body")
    df = pd.read_csv(io.StringIO(csv_text))
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    # Drop any non-numeric columns (safety)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("Dataset has no numeric columns")
    return numeric


def _load_lstm_dataframe(body, target_column):
    csv_text = body.get("csv")
    if not csv_text:
        raise ValueError("Missing 'csv' in request body")
    df = pd.read_csv(io.StringIO(csv_text))
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    numeric = df.select_dtypes(include=[np.number])
    if target_column and target_column not in numeric.columns:
        raise ValueError(f"Target column '{target_column}' not found or not numeric")
    return numeric


def _get_lstm_model(side, model_id, model_b64):
    if model_b64:
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp_path = f.name
            f.write(base64.b64decode(model_b64))
        try:
            model, scaler, columns, input_dim = lstm.load_model(tmp_path)
        finally:
            os.unlink(tmp_path)
        return model, scaler, columns, input_dim
    key = (side, model_id)
    if key not in _LSTM_MODELS:
        raise ValueError(f"No trained model found for side={side}. Train first.")
    entry = _LSTM_MODELS[key]
    return entry['model'], entry['scaler'], entry['columns'], entry['input_dim']


def _seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@app.post("/api/generate/emcm-ps")
def generate_emcm_ps():
    try:
        body = request.get_json(force=True) or {}
        params = body.get("params", {})
        data = _load_dataframe(body)

        num_steps              = int(params.get("num_steps", 2000))
        num_augmented_samples  = int(params.get("num_augmented_samples", 200))
        noise_level            = float(params.get("noise_level", 0.03))
        num_buckets            = int(params.get("num_buckets", 5))
        decoder_mode           = str(params.get("decoder_mode", "local"))
        start_row              = int(params.get("start_row", 0))
        end_row                = params.get("end_row")

        if end_row is not None and end_row != "":
            data = data.iloc[start_row:int(end_row)].reset_index(drop=True)
        elif start_row:
            data = data.iloc[start_row:].reset_index(drop=True)

        _seed()

        # Augmentation
        aug = emcm_ps.augment_data(
            data,
            num_new_samples=num_augmented_samples,
            noise_level=noise_level,
        )

        # Discretization
        disc, edges = emcm_ps.adaptive_discretize(aug, num_buckets=num_buckets)

        state_ids, state_mapping = emcm_ps.define_states(disc)
        num_states = len(state_mapping)
        state_to_rows = emcm_ps.build_state_to_rows_map(state_ids, aug)

        # Probabilistic state sampling
        counts = Counter(state_ids)
        raw = np.array([counts[i] for i in range(num_states)], dtype=float)
        probs = raw / raw.sum()
        generated_sequence = np.random.choice(num_states, size=num_steps, p=probs)

        out = emcm_ps.generate_continuous_data_joint(
            generated_sequence, state_mapping, edges,
            augmented_data=aug,
            state_to_rows_map=state_to_rows,
            decoder_mode=decoder_mode,
        )

        stats = {
            "num_unique_states": int(num_states),
            "state_ratio": round(num_states / len(state_ids), 4),
            "num_augmented_rows": int(len(aug)),
            "num_buckets_per_col": {c: int(len(e) - 1) for c, e in edges.items()},
        }

        return jsonify({
            "columns": out.columns.tolist(),
            "rows": out.values.tolist(),
            "stats": stats,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# ── LSTM endpoints ───────────────────────────────────────────────────────────

def _align_to_columns(df, train_columns, target_column):
    if target_column not in df.columns:
        avail = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Target column '{target_column}' not found in CSV. Available: {avail}"
        )
    train_features = [c for c in train_columns if c != target_column]
    df_features    = [c for c in df.columns if c != target_column]
    missing        = [c for c in train_features if c not in df_features]
    extras         = [c for c in df_features if c not in train_features]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Required: {train_features + [target_column]}."
        )
    ordered  = train_features + [target_column]
    warnings = []
    if extras:
        warnings.append(f"Extra columns ignored: {extras}")
    return df[ordered], warnings


@app.post("/api/lstm/train")
def lstm_train():
    try:
        body          = request.get_json(force=True) or {}
        params        = body.get("params", {})
        target_column = body.get("target_column", "")
        side          = body.get("side", "left")

        df = _load_lstm_dataframe(body, target_column)
        total_rows = len(df)

        epochs      = int(params.get("epochs", 50))
        batch_size  = int(params.get("batch_size", 32))
        lr          = float(params.get("learning_rate", 0.003))
        patience    = int(params.get("early_stopping_patience", 10))
        contam      = float(params.get("contamination", 0.05))
        train_ratio = float(params.get("train_ratio", 0.8))
        apply_iso_test = bool(params.get("apply_outlier_removal_test", False))

        # Split before preprocessing so outlier filtering on test is controlled independently of train
        split    = int(train_ratio * len(df))
        df_train = df.iloc[:split].reset_index(drop=True)
        df_test  = df.iloc[split:].reset_index(drop=True)

        data_train, columns, scaler, input_dim = lstm.preprocess_dataframe(
            df_train, target_column, contam, apply_outlier_removal=True
        )

        target_idx = columns.index(target_column)
        X_train = data_train[:, :input_dim]
        y_train = data_train[:, target_idx]

        if len(df_test) > 0:
            data_test, _, _, _ = lstm.preprocess_dataframe(
                df_test, target_column, contam,
                scaler=scaler, apply_outlier_removal=apply_iso_test,
            )
            X_test = data_test[:, :input_dim]
            y_test = data_test[:, target_idx]
        else:
            X_test = np.empty((0, input_dim))
            y_test = np.empty((0,))

        model = lstm.LSTMModel(input_dim).to(lstm.device)
        model = lstm.train_model(model, X_train, y_train, epochs, batch_size, lr, patience)

        train_pred = lstm.predict(model, X_train)
        test_pred  = lstm.predict(model, X_test) if len(X_test) > 0 else np.array([])

        def rescale(Xp, yp):
            if len(Xp) == 0:
                return np.array([])
            return lstm.inverse_transform(scaler, np.column_stack((Xp, yp)))[:, -1]

        train_true_r = rescale(X_train, y_train).tolist()
        train_pred_r = rescale(X_train, train_pred).tolist()
        test_true_r  = rescale(X_test,  y_test).tolist()
        test_pred_r  = rescale(X_test,  test_pred).tolist()

        model_id = str(_uuid.uuid4())[:8]
        _LSTM_MODELS[(side, model_id)] = {
            'model': model, 'scaler': scaler,
            'columns': columns, 'input_dim': input_dim,
        }

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp_path = f.name
        try:
            lstm.save_model(model, scaler, columns, input_dim, tmp_path)
            with open(tmp_path, 'rb') as f:
                model_b64 = base64.b64encode(f.read()).decode()
        finally:
            os.unlink(tmp_path)

        return jsonify({
            'columns':    columns,
            'train':      {'y_true': train_true_r, 'y_pred': train_pred_r},
            'test':       {'y_true': test_true_r,  'y_pred': test_pred_r},
            'model_id':   model_id,
            'total_rows': total_rows,
            'train_rows': len(train_true_r),
            'test_rows':  len(test_true_r),
            'model_b64':  model_b64,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.post("/api/lstm/validate")
def lstm_validate():
    try:
        body          = request.get_json(force=True) or {}
        target_column = body.get("target_column", "")
        side          = body.get("side", "left")
        model_id      = body.get("model_id")
        model_b64     = body.get("model_b64")
        contam        = float(body.get("contamination", 0.05))
        apply_iso     = bool(body.get("apply_outlier_removal_test", False))

        model, scaler, columns, input_dim = _get_lstm_model(side, model_id, model_b64)

        df = _load_lstm_dataframe(body, target_column)
        df_aligned, col_warnings = _align_to_columns(df, columns, target_column)

        data, _, _, _ = lstm.preprocess_dataframe(
            df_aligned, target_column, contam,
            scaler=scaler, apply_outlier_removal=apply_iso,
        )

        target_idx = columns.index(target_column)
        X = data[:, :input_dim]
        y = data[:, target_idx]

        pred = lstm.predict(model, X) if len(X) > 0 else np.array([])

        def rescale(Xp, yp):
            if len(Xp) == 0:
                return np.array([])
            return lstm.inverse_transform(scaler, np.column_stack((Xp, yp)))[:, -1]

        y_true_r = rescale(X, y).tolist()
        y_pred_r = rescale(X, pred).tolist()

        return jsonify({
            'y_true':       y_true_r,
            'y_pred':       y_pred_r,
            'valid_rows':   len(y_true_r),
            'col_warnings': col_warnings,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.post("/api/lstm/predict")
def lstm_predict():
    try:
        body         = request.get_json(force=True) or {}
        target_column = body.get("target_column", "")
        side         = body.get("side", "left")
        model_id     = body.get("model_id")
        model_b64_in = body.get("model_b64")
        csv_rows     = body.get("csv_rows", [])
        row_columns  = body.get("columns", [])

        model, scaler, columns, input_dim = _get_lstm_model(side, model_id, model_b64_in)

        df = pd.DataFrame(csv_rows, columns=row_columns)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # csv_rows contains only feature columns — normalise using training scaler directly
        mean, std = scaler
        mean_np = mean.cpu().numpy() if hasattr(mean, 'cpu') else np.array(mean)
        std_np  = std.cpu().numpy()  if hasattr(std,  'cpu') else np.array(std)

        X_raw  = df.values.astype(np.float32)
        X_norm = (X_raw - mean_np[0, :input_dim]) / std_np[0, :input_dim]

        pred_norm    = lstm.predict(model, X_norm)
        predictions  = (pred_norm * std_np[0, input_dim] + mean_np[0, input_dim]).tolist()

        return jsonify({'predictions': predictions})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# ── Lifelong-LSTM endpoints ──────────────────────────────────────────────────

def _lifelong_align(source_df, target_df_raw, target_column):
    if target_column not in target_df_raw.columns:
        avail = ", ".join(target_df_raw.columns.tolist())
        raise ValueError(
            f"Target column '{target_column}' not found in target CSV. "
            f"Available: {avail}"
        )
    source_features = [c for c in source_df.columns if c != target_column]
    target_features = [c for c in target_df_raw.columns if c != target_column]
    missing_cols    = [c for c in source_features if c not in target_features]
    extra_cols      = [c for c in target_features if c not in source_features]

    if missing_cols:
        raise ValueError(
            f"Target CSV is missing required columns: {missing_cols}. "
            f"Required: {source_features + [target_column]}."
        )

    ordered_cols = source_features + [target_column]
    warnings     = []
    if extra_cols:
        warnings.append(f"Extra columns in target (ignored): {extra_cols}")
    return source_df[ordered_cols], target_df_raw[ordered_cols], warnings


@app.post("/api/lifelong/optimize")
def lifelong_optimize():
    try:
        body          = request.get_json(force=True) or {}
        params        = body.get("params", {})
        target_column = body.get("target_column", "")

        df = _load_lstm_dataframe(body, target_column)
        _seed()

        contamination = float(params.get("contamination", lifelong_lstm.DEFAULTS["contamination"]))
        train_ratio   = float(params.get("train_ratio",   lifelong_lstm.DEFAULTS["train_ratio"]))
        init_points   = int(params.get("bayes_init_points", lifelong_lstm.DEFAULTS["bayes_init_points"]))
        n_iter        = int(params.get("bayes_n_iter",       lifelong_lstm.DEFAULTS["bayes_n_iter"]))

        data, columns, scaler, input_dim = lifelong_lstm.preprocess_data(df, target_column, contamination)

        X = data[:, :input_dim]
        y = data[:, input_dim]
        split    = int(train_ratio * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]

        result = lifelong_lstm.bayes_optimize(
            X_train, y_train, X_test, y_test, input_dim, params, init_points, n_iter
        )
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.post("/api/lifelong/train")
def lifelong_train():
    try:
        body          = request.get_json(force=True) or {}
        params        = body.get("params", {})
        target_column = body.get("target_column", "")
        side          = body.get("side", "left")

        source_csv = body.get("source_csv") or body.get("csv", "")
        source_df  = _load_lstm_dataframe({"csv": source_csv}, target_column)
        _seed()

        contamination = float(params.get("contamination", lifelong_lstm.DEFAULTS["contamination"]))

        data1, columns, scaler, input_dim = lifelong_lstm.preprocess_data(
            source_df, target_column, contamination
        )

        result     = lifelong_lstm.run_train(data1, input_dim, scaler, params)
        streaming  = result.pop("_streaming")
        scaler_out = result.pop("_scaler")
        inp_dim    = result.pop("_input_dim")

        session_id = str(_uuid.uuid4())[:8]
        if len(_LIFELONG_SESSIONS) >= 8:
            oldest = next(iter(_LIFELONG_SESSIONS))
            del _LIFELONG_SESSIONS[oldest]
        _LIFELONG_SESSIONS[(side, session_id)] = {
            "streaming_snapshot": copy.deepcopy(streaming),
            "scaler":             scaler_out,
            "input_dim":          inp_dim,
            "source_cols":        columns,
            "target_column":      target_column,
        }

        result["session_id"] = session_id
        result["columns"]    = columns
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.post("/api/lifelong/adapt")
def lifelong_adapt():
    try:
        body          = request.get_json(force=True) or {}
        params        = body.get("params", {})
        session_id    = body.get("session_id")
        side          = body.get("side", "left")
        target_csv    = body.get("target_csv", "")

        key = (side, session_id)
        if key not in _LIFELONG_SESSIONS:
            raise ValueError(f"No trained session for side={side}. Run Train first.")

        entry         = _LIFELONG_SESSIONS[key]
        scaler        = entry["scaler"]
        input_dim     = entry["input_dim"]
        source_cols   = entry["source_cols"]
        target_column = entry["target_column"]

        source_skeleton = pd.DataFrame(columns=source_cols)
        target_df_raw   = _load_lstm_dataframe({"csv": target_csv}, "")
        _, target_df, col_warnings = _lifelong_align(source_skeleton, target_df_raw, target_column)

        contamination = float(params.get("contamination", lifelong_lstm.DEFAULTS["contamination"]))

        data2, _, _, _ = lifelong_lstm.preprocess_data(
            target_df, target_column, contamination, scaler=scaler
        )

        streaming = copy.deepcopy(entry["streaming_snapshot"])
        result = lifelong_lstm.run_adapt(streaming, data2, input_dim, scaler, params)
        result["col_warnings"] = col_warnings
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Serving Synthetic Data Comparison on http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
