import heapq
import os
import numpy as np
import pandas as pd
import random
from collections import Counter


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


# ============================================================================
# STAGE 0 — Data Loading
# ============================================================================

def load_data(file_path, start_row=0, end_row=None):
    data = pd.read_csv(file_path, skiprows=range(1, start_row + 1), nrows=(end_row - start_row if end_row else None))
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    return data


# ============================================================================
# STAGE 1 — Augmentation (correlation-aware multiplicative noise, round-robin)
# ============================================================================

def augment_data(data, num_new_samples=100, noise_level=0.05):
    n_rows, n_features = data.shape

    real_corr = np.nan_to_num(data.corr().values, nan=0.0)
    np.fill_diagonal(real_corr, 1.0)
    cov_matrix = real_corr * (noise_level ** 2) + np.eye(n_features) * 1e-8

    # Round-robin over real rows so every row gets equal representation.
    row_idx = np.arange(num_new_samples) % n_rows
    noise = np.random.multivariate_normal(
        np.zeros(n_features), cov_matrix, size=num_new_samples)
    augmented = np.maximum(data.values[row_idx] * (1 + noise), 0)

    augmented_df = pd.DataFrame(augmented, columns=data.columns)
    return pd.concat([data, augmented_df], ignore_index=True)


# ============================================================================
# STAGE 2 — Discretization
#   a) adaptive_discretize            — fixed quantile bins (fallback)
#   b) adaptive_bucketing_discretize  — cluster + T-quantile sub-bins, binary-search T
# ============================================================================

def adaptive_discretize(data, num_buckets=5):
    discretized_data = {}
    bucket_edges_dict = {}

    for column in data.columns:
        min_val = data[column].min()
        max_val = data[column].max()

        if min_val == max_val:
            bucket_edges = np.array([min_val, min_val + EDGE_EPSILON])
            bucket_labels = np.zeros_like(data[column], dtype=int)
        else:
            bucket_edges = np.unique(np.quantile(data[column], np.linspace(0, 1, num_buckets + 1)))
            if len(bucket_edges) < 2:
                bucket_edges = np.linspace(min_val, max_val, num_buckets + 1)
            bucket_labels = np.digitize(data[column], bucket_edges) - 1
            bucket_labels = np.clip(bucket_labels, 0, len(bucket_edges) - 2)

        discretized_data[column] = bucket_labels
        bucket_edges_dict[column] = bucket_edges

    discretized_df = pd.DataFrame(discretized_data)
    return discretized_df, bucket_edges_dict


EDGE_EPSILON = 1e-9
MAX_SPLITS = 2000


def _digitize(values, edges):
    return np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)


def adaptive_bucketing_discretize(data, max_state_ratio=0.3, max_buckets_per_feature=1000,
                        noise_level=0.05, return_summary=False):
    n_total = len(data)
    cols = list(data.columns)

    edges = {}
    col_buckets = {}
    for col in cols:
        v = data[col].values.astype(float)
        lo, hi = float(v.min()), float(v.max())
        edges[col] = [lo, hi + EDGE_EPSILON]
        col_buckets[col] = np.zeros(n_total, dtype=int)

    def current_ratio(overrides=None):
        arrs = [overrides[c] if overrides and c in overrides else col_buckets[c] for c in cols]
        rows = set(map(tuple, np.stack(arrs, axis=1).tolist()))
        return len(rows) / n_total

    heap = []
    seen = set()
    for col in cols:
        uniq = np.sort(np.unique(data[col].values.astype(float)))
        rng = float(uniq[-1] - uniq[0]) or 1.0
        for i in range(len(uniq) - 1):
            g = float(uniq[i+1] - uniq[i])
            if g > EDGE_EPSILON:
                heapq.heappush(heap, (-g / rng, col, float(uniq[i]), float(uniq[i+1]), g))

    step = 0
    accepted = 0
    while heap and step < MAX_SPLITS:
        neg_rel, col, lv, rv, abs_gap = heapq.heappop(heap)
        key = (col, round(lv, 10), round(rv, 10))
        if key in seen:
            continue
        seen.add(key)
        if len(edges[col]) - 1 >= max_buckets_per_feature:
            continue

        trial_e = sorted(set(edges[col] + [lv + EDGE_EPSILON, rv - EDGE_EPSILON]))
        trial_labels = _digitize(data[col].values.astype(float), trial_e)
        trial_ratio = current_ratio({col: trial_labels})

        step += 1
        if trial_ratio <= max_state_ratio:
            edges[col] = trial_e
            col_buckets[col] = trial_labels
            accepted += 1

    for col in cols:
        edges[col] = np.asarray(edges[col], dtype=float)

    disc = pd.DataFrame(col_buckets)
    final_ratio = current_ratio()
    unreachable = final_ratio > max_state_ratio
    edges_out = {c: np.asarray(edges[c], dtype=float) for c in cols}

    header = "target unreachable" if unreachable else "converged"
    print(f"\n--- STAGE 2: Adaptive Bucketing (recursive largest-gap, {header}) ---")
    print(f"  Splits: {step}, accepted: {accepted}, state ratio: {final_ratio:.3f} (target <= {max_state_ratio})")
    for c in cols:
        e = edges_out[c]
        used = len(set(disc[c].tolist()))
        empty = (len(e) - 1) - used
        tag = f", {empty} empty" if empty else ""
        print(f"  {c}: {len(e)-1} buckets{tag} -> {['%.4g' % v for v in e]}")

    if return_summary:
        summary = {
            'phase': 'final',
            'n_splits': step,
            'n_accepted': accepted,
            'final_ratio': round(float(final_ratio), 4),
            'target_ratio': float(max_state_ratio),
            'unreachable': bool(unreachable),
            'final_edges': {c: [float(v) for v in edges_out[c]] for c in cols},
        }
        return disc, edges_out, summary
    return disc, edges_out


# ============================================================================
# STAGE 3 — State Encoding
# ============================================================================

def define_states(discretized_df):
    state_tuples = discretized_df.to_numpy()
    state_strings = np.array(['_'.join(map(str, state)) for state in state_tuples])
    state_ids, unique_states = pd.factorize(state_strings)
    state_mapping = {state: idx for idx, state in enumerate(unique_states)}
    return state_ids, state_mapping


def build_state_to_rows_map(state_ids, augmented_data):
    state_to_rows = {}
    for i, sid in enumerate(state_ids):
        if sid not in state_to_rows:
            state_to_rows[sid] = []
        state_to_rows[sid].append(i)
    return state_to_rows


# ============================================================================
# STAGE 5 — Decoding
#   a) map_joint_state_to_continuous_values — uniform within bucket
#   b) decode_state_local_sampling          — local multivariate-normal
#   c) generate_continuous_data_joint       — top-level driver
# ============================================================================

def map_joint_state_to_continuous_values(state_tuple, bucket_edges_dict):
    parts = [int(float(x)) for x in state_tuple.split('_')]
    continuous_values = []

    for feature_idx, feature in enumerate(bucket_edges_dict.keys()):
        state = parts[feature_idx]
        edges = bucket_edges_dict[feature]
        if state + 1 >= len(edges):
            min_val = edges[-2]
            max_val = edges[-1]
        else:
            min_val = edges[state]
            max_val = edges[state + 1]
        continuous_value = min_val + np.random.random() * (max_val - min_val)
        continuous_values.append(continuous_value)
    return continuous_values


def _get_bucket_bounds(state_tuple, bucket_edges_dict):
    parts = [int(float(x)) for x in state_tuple.split('_')]
    features = list(bucket_edges_dict.keys())
    lower = np.empty(len(features))
    upper = np.empty(len(features))
    for i, feature in enumerate(features):
        bucket_idx = parts[i]
        edges = bucket_edges_dict[feature]
        if bucket_idx + 1 >= len(edges):
            lower[i] = edges[-2]
            upper[i] = edges[-1]
        else:
            lower[i] = edges[bucket_idx]
            upper[i] = edges[bucket_idx + 1]
    return lower, upper


def decode_state_local_sampling(state_id, state_to_rows_map, augmented_data,
                                bucket_edges_dict, reverse_mapping):
    row_indices = state_to_rows_map.get(state_id, [])
    features = list(bucket_edges_dict.keys())
    n_features = len(features)

    state_tuple = reverse_mapping[state_id]
    lower_bounds, upper_bounds = _get_bucket_bounds(state_tuple, bucket_edges_dict)

    if len(row_indices) >= 3:
        local_data = augmented_data.iloc[row_indices][features].values
        mean = local_data.mean(axis=0)
        cov = np.cov(local_data, rowvar=False)
        cov += np.eye(n_features) * 1e-8
        sample = np.random.multivariate_normal(mean, cov)
        sample = np.clip(sample, lower_bounds, upper_bounds)
        return np.maximum(sample, 0).tolist()

    elif len(row_indices) == 2:
        row1 = augmented_data.iloc[row_indices[0]][features].values.astype(float)
        row2 = augmented_data.iloc[row_indices[1]][features].values.astype(float)
        weight = np.random.random()
        sample = weight * row1 + (1 - weight) * row2
        sample = np.clip(sample, lower_bounds, upper_bounds)
        return np.maximum(sample, 0).tolist()

    else:
        sample = lower_bounds + np.random.random(n_features) * (upper_bounds - lower_bounds)
        return np.maximum(sample, 0).tolist()


DECODER_MODES = ("local", "uniform")


def generate_continuous_data_joint(state_sequence, state_mapping, bucket_edges_dict,
                                   augmented_data=None, state_to_rows_map=None,
                                   decoder_mode="uniform"):
    if decoder_mode not in DECODER_MODES:
        raise ValueError(f"decoder_mode must be one of {DECODER_MODES}, got {decoder_mode!r}")

    use_local = (
        decoder_mode == "local"
        and augmented_data is not None
        and state_to_rows_map is not None
    )
    reverse_mapping = {v: k for k, v in state_mapping.items()}
    continuous_data = []

    for state_id in state_sequence:
        if use_local:
            continuous_values = decode_state_local_sampling(
                state_id, state_to_rows_map, augmented_data,
                bucket_edges_dict, reverse_mapping)
        else:
            state_tuple = reverse_mapping[state_id]
            continuous_values = map_joint_state_to_continuous_values(state_tuple, bucket_edges_dict)
        continuous_data.append(continuous_values)

    cols = list(bucket_edges_dict.keys())
    continuous_df = pd.DataFrame(continuous_data, columns=cols)
    return continuous_df


# ============================================================================
# MAIN — Pipeline driver
# ============================================================================

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ── Paths ──────────────────────────────────────────────────────────────
    file_path = os.path.join(SCRIPT_DIR, "data", "augmented_effluent_data_KT_AnMBR_Bac.csv")
    output_path = os.path.join(SCRIPT_DIR, "output", "all4changes-effluent_data_KT_AnMBR_Bac.csv")

    # ── Parameters ─────────────────────────────────────────────────────────
    num_steps = 2000
    num_augmented_samples = 1
    noise_level = 0.05
    num_buckets = 5
    adaptive_bucketing = False
    max_state_ratio = 0.3
    max_buckets_per_feature = 1000
    decoder_mode = "uniform"

    # ── Stage 0: Load ──────────────────────────────────────────────────────
    data = load_data(file_path)
    print("\n--- STAGE 0: Original Data (First 3 Rows) ---")
    print(data.head(3))

    # ── Stage 1: Augment ───────────────────────────────────────────────────
    augmented_data = augment_data(data, num_new_samples=num_augmented_samples, noise_level=noise_level)
    print(f"\n--- STAGE 1: Multivariate Noise Augmentation (Last 3 Rows) ---")
    print(augmented_data.tail(3))

    # ── Stage 2: Discretize ────────────────────────────────────────────────
    if adaptive_bucketing:
        discretized_data, bucket_edges = adaptive_bucketing_discretize(
            augmented_data, max_state_ratio=max_state_ratio,
            max_buckets_per_feature=max_buckets_per_feature, noise_level=noise_level)
    else:
        discretized_data, bucket_edges = adaptive_discretize(augmented_data, num_buckets=num_buckets)
        print(f"\n--- STAGE 2: Data Discretized (quantile-based, {num_buckets} buckets) ---")
        for col, edges in bucket_edges.items():
            print(f"  {col}: {len(edges)-1} buckets -> {['%.4f' % e for e in edges]}")

    # ── Stage 3: Encode states ─────────────────────────────────────────────
    state_ids, state_mapping = define_states(discretized_data)
    num_states = len(state_mapping)
    total_rows = len(state_ids)
    print(f"\n--- STAGE 3: State Encoding ---")
    print(f"Unique states: {num_states} / {total_rows} rows (ratio: {num_states/total_rows:.2f})")

    inv_map = {v: k for k, v in state_mapping.items()}

    all_states_df = pd.DataFrame([
        {"Row_Index": i, "State_ID": state_ids[i], "State_Code": inv_map[state_ids[i]]}
        for i in range(len(state_ids))
    ])

    counts = Counter(state_ids)
    unique_states_df = pd.DataFrame([
        {
            "State_ID": state_id,
            "State_Code": state_code,
            "Count": counts[state_id],
            "Prob_Pct": round((counts[state_id] / total_rows) * 100, 2),
        }
        for state_code, state_id in state_mapping.items()
    ]).sort_values("Prob_Pct", ascending=False)

    state_to_rows_map = build_state_to_rows_map(state_ids, augmented_data)

    rows_per_state = [len(rows) for rows in state_to_rows_map.values()]
    one_row = sum(1 for r in rows_per_state if r == 1)
    two_three = sum(1 for r in rows_per_state if 2 <= r <= 3)
    four_plus = sum(1 for r in rows_per_state if r >= 4)

    # ── Stage 4: Probabilistic sampling ────────────────────────────────────
    raw_probs = unique_states_df["Prob_Pct"].values / 100
    state_probs = raw_probs / raw_probs.sum()
    generated_sequence = np.random.choice(unique_states_df["State_ID"], size=num_steps, p=state_probs)
    print("\n--- STAGE 4: Probabilistic Sampling ---")
    print(f"First 10 Synthetic State IDs: {generated_sequence[:10]}")

    # ── Stage 5: Decode ────────────────────────────────────────────────────
    decoder_name = {"local": "Local Distribution Sampling", "uniform": "Uniform Random"}[decoder_mode]
    generated_continuous_data = generate_continuous_data_joint(
        generated_sequence, state_mapping, bucket_edges,
        augmented_data=augmented_data,
        state_to_rows_map=state_to_rows_map,
        decoder_mode=decoder_mode)
    print(f"\n--- STAGE 5: Decoding ({decoder_name}) ---")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generated_continuous_data.to_csv(output_path, index=False)
    print(f"\nDecoded/Generated data has been saved to '{output_path}'")


if __name__ == "__main__":
    main()
