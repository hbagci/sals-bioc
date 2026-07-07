import heapq
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import warnings
from copulas.multivariate import GaussianMultivariate

random_seed = 42
np.random.seed(random_seed)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Adaptive bucketing constants ─────────────────
_EDGE_EPSILON = 1e-9
_MAX_SPLITS   = 2000


def _ab_digitize(values, edges):
    return np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)

class ImprovedMarkovGenerator:

    def __init__(self, num_buckets=10, smoothing_alpha=0.01, use_kde=True,
                 adaptive_bucketing=False, max_state_ratio=0.3,
                 max_buckets_per_feature=1000):
        self.num_buckets = num_buckets
        self.smoothing_alpha = smoothing_alpha
        self.use_kde = use_kde
        self.adaptive_bucketing = adaptive_bucketing
        self.max_state_ratio = max_state_ratio
        self.max_buckets_per_feature = max_buckets_per_feature
        self.kde_estimators = {}
        self.bucket_distributions = {}

    def load_data(self, file_path, start_row=0, end_row=None):
        data = pd.read_csv(file_path, skiprows=range(1, start_row + 1),
                          nrows=(end_row - start_row if end_row else None))
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
        return data

    def adaptive_discretization(self, data):
        discretized_data = {}
        bucket_edges_dict = {}

        for column in data.columns:
            col_data = data[column].values

            if len(np.unique(col_data)) <= self.num_buckets:
                bucket_edges = np.sort(np.unique(col_data))
                bucket_labels = np.searchsorted(bucket_edges[:-1], col_data)
            else:
                quantiles = np.linspace(0, 100, self.num_buckets + 1)
                bucket_edges = np.percentile(col_data, quantiles)
                bucket_edges = np.unique(bucket_edges)

                # Slightly expand boundaries to include all data
                bucket_edges[0] -= 1e-10
                bucket_edges[-1] += 1e-10

                bucket_labels = np.digitize(col_data, bucket_edges) - 1
                bucket_labels = np.clip(bucket_labels, 0, len(bucket_edges) - 2)

            discretized_data[column] = bucket_labels
            bucket_edges_dict[column] = bucket_edges

            if self.use_kde:
                self.bucket_distributions[column] = {}
                for bucket_id in range(len(bucket_edges) - 1):
                    bucket_mask = (bucket_labels == bucket_id)
                    bucket_count = np.sum(bucket_mask)

                    if bucket_count > 0:
                        bucket_values = col_data[bucket_mask]

                        kde_success = False
                        if bucket_count > 5 and np.std(bucket_values) > 1e-10:
                            try:
                                # Add small noise to avoid singular matrix
                                values_with_noise = bucket_values + np.random.normal(0, 1e-8, len(bucket_values))
                                kde = gaussian_kde(values_with_noise, bw_method='scott')
                                kde_success = True
                            except Exception as e:
                                print(f"KDE failed for {column}, bucket {bucket_id}: {e}")
                                kde_success = False

                        dist_info = {
                            'min': bucket_values.min(),
                            'max': bucket_values.max(),
                            'values': bucket_values,
                            'mean': bucket_values.mean(),
                            'std': bucket_values.std() if bucket_count > 1 else 0
                        }

                        if kde_success:
                            dist_info['kde'] = kde

                        self.bucket_distributions[column][bucket_id] = dist_info

        return pd.DataFrame(discretized_data), bucket_edges_dict

    def adaptive_bucketing_discretize(self, data, return_trace=False):
        n_total = len(data)
        cols = list(data.columns)

        edges = {}
        col_buckets = {}
        for col in cols:
            v = data[col].values.astype(float)
            lo, hi = float(v.min()), float(v.max())
            edges[col] = [lo, hi + _EDGE_EPSILON]
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
                g = float(uniq[i + 1] - uniq[i])
                if g > _EDGE_EPSILON:
                    heapq.heappush(heap, (-g / rng, col, float(uniq[i]), float(uniq[i + 1]), g))

        step = 0
        n_accepted = 0
        while heap and step < _MAX_SPLITS:
            neg_rel, col, lv, rv, abs_gap = heapq.heappop(heap)
            key = (col, round(lv, 10), round(rv, 10))
            if key in seen:
                continue
            seen.add(key)
            if len(edges[col]) - 1 >= self.max_buckets_per_feature:
                continue

            trial_e = sorted(set(edges[col] + [lv + _EDGE_EPSILON, rv - _EDGE_EPSILON]))
            trial_labels = _ab_digitize(data[col].values.astype(float), trial_e)
            trial_ratio = current_ratio({col: trial_labels})

            step += 1
            if trial_ratio <= self.max_state_ratio:
                edges[col] = trial_e
                col_buckets[col] = trial_labels
                n_accepted += 1

        for col in cols:
            edges[col] = np.asarray(edges[col], dtype=float)

        final_ratio = current_ratio()
        header = "target unreachable" if final_ratio > self.max_state_ratio else "converged"
        print(f"\n--- Adaptive Bucketing (largest-gap, {header}) ---")
        print(f"  Splits: {step}, state ratio: {final_ratio:.3f} (target <= {self.max_state_ratio})")
        for c in cols:
            e = edges[c]
            used = len(set(col_buckets[c].tolist()))
            empty = (len(e) - 1) - used
            tag = f", {empty} empty" if empty else ""
            print(f"  {c}: {len(e)-1} buckets{tag}")

        bucket_edges_dict = {c: edges[c] for c in cols}
        discretized_data = {c: col_buckets[c] for c in cols}

        if self.use_kde:
            self.bucket_distributions = {}
            for col in cols:
                col_data = data[col].values
                bucket_labels = col_buckets[col]
                bucket_edges = edges[col]
                self.bucket_distributions[col] = {}
                for bucket_id in range(len(bucket_edges) - 1):
                    bucket_mask = (bucket_labels == bucket_id)
                    bucket_count = int(np.sum(bucket_mask))
                    if bucket_count == 0:
                        continue
                    bucket_values = col_data[bucket_mask]
                    dist_info = {
                        'min': bucket_values.min(),
                        'max': bucket_values.max(),
                        'values': bucket_values,
                        'mean': bucket_values.mean(),
                        'std': bucket_values.std() if bucket_count > 1 else 0,
                    }
                    if bucket_count > 5 and np.std(bucket_values) > 1e-10:
                        try:
                            values_with_noise = bucket_values + np.random.normal(0, 1e-8, len(bucket_values))
                            dist_info['kde'] = gaussian_kde(values_with_noise, bw_method='scott')
                        except Exception:
                            pass
                    self.bucket_distributions[col][bucket_id] = dist_info

        disc_df = pd.DataFrame(discretized_data)
        unreachable = final_ratio > self.max_state_ratio
        if return_trace:
            summary = {
                'phase': 'final',
                'n_splits': step,
                'n_accepted': n_accepted,
                'final_ratio': round(float(final_ratio), 4),
                'target_ratio': float(self.max_state_ratio),
                'unreachable': bool(unreachable),
                'final_edges': {c: [float(v) for v in edges[c]] for c in cols},
            }
            return disc_df, bucket_edges_dict, [summary]
        return disc_df, bucket_edges_dict

    def build_smoothed_transition_matrix(self, state_ids, num_states):
        # Initialize with smoothing values instead of zeros so no transition has zero probability
        transition_matrix = np.ones((num_states, num_states)) * self.smoothing_alpha

        for i in range(len(state_ids) - 1):
            current_state = state_ids[i]
            next_state = state_ids[i + 1]
            if current_state < num_states and next_state < num_states:
                transition_matrix[current_state, next_state] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / row_sums

        return transition_matrix

    def generate_diverse_sequences(self, transition_matrix, num_sequences,
                                  steps_per_sequence, state_frequencies):
        all_sequences = []

        for _ in range(num_sequences):
            start_state = np.random.choice(len(state_frequencies),
                                         p=state_frequencies)

            sequence = self.random_walk(transition_matrix, start_state,
                                       steps_per_sequence)
            all_sequences.extend(sequence)

        return all_sequences

    def random_walk(self, transition_matrix, start_state, num_steps):
        current_state = start_state
        state_sequence = [current_state]

        for _ in range(num_steps - 1):
            probabilities = transition_matrix[current_state]
            next_state = np.random.choice(len(transition_matrix), p=probabilities)
            state_sequence.append(next_state)
            current_state = next_state

        return state_sequence

    def map_to_continuous_with_kde(self, state_tuple, bucket_edges_dict):
        continuous_values = []
        state_list = [int(x) for x in state_tuple.split('_')]

        for feature_idx, feature in enumerate(bucket_edges_dict.keys()):
            bucket_id = state_list[feature_idx]

            if self.use_kde and feature in self.bucket_distributions:
                if bucket_id in self.bucket_distributions[feature]:
                    dist_info = self.bucket_distributions[feature][bucket_id]

                    if 'kde' in dist_info:
                        try:
                            sample = dist_info['kde'].resample(1)[0][0]
                            sample = np.clip(sample, dist_info['min'], dist_info['max'])
                        except:
                            if len(dist_info['values']) > 1:
                                base_value = np.random.choice(dist_info['values'])
                                noise = np.random.normal(0, dist_info['std'] * 0.1) if dist_info['std'] > 0 else 0
                                sample = base_value + noise
                                sample = np.clip(sample, dist_info['min'], dist_info['max'])
                            else:
                                sample = dist_info['values'][0]
                    elif len(dist_info['values']) > 0:
                        base_value = np.random.choice(dist_info['values'])
                        if dist_info['std'] > 0 and len(dist_info['values']) > 1:
                            noise = np.random.normal(0, dist_info['std'] * 0.1)
                            sample = base_value + noise
                            sample = np.clip(sample, dist_info['min'], dist_info['max'])
                        else:
                            sample = base_value
                    else:
                        sample = self.uniform_sample(bucket_edges_dict[feature], bucket_id)

                    continuous_values.append(sample)
                else:
                    continuous_values.append(self.uniform_sample(
                        bucket_edges_dict[feature], bucket_id))
            else:
                continuous_values.append(self.uniform_sample(
                    bucket_edges_dict[feature], bucket_id))

        return continuous_values

    def uniform_sample(self, edges, bucket_id):
        if bucket_id + 1 >= len(edges):
            min_val = edges[bucket_id]
            max_val = edges[-1]
        else:
            min_val = edges[bucket_id]
            max_val = edges[bucket_id + 1]
        return min_val + np.random.random() * (max_val - min_val)

    def add_gaussian_noise(self, data, noise_scale=0.02):
        noisy_data = data.copy()
        for column in data.columns:
            std = data[column].std()
            noise = np.random.normal(0, std * noise_scale, len(data))
            noisy_data[column] += noise
        return noisy_data

    def generate(self, data, num_samples=2000, add_noise=True):
        print("Step 1: Discretization...")
        self.last_bucketing_trace = None
        if self.adaptive_bucketing:
            discretized_data, bucket_edges, self.last_bucketing_trace = self.adaptive_bucketing_discretize(data, return_trace=True)
        else:
            discretized_data, bucket_edges = self.adaptive_discretization(data)

        print("Step 2: Building state mapping...")
        state_tuples = discretized_data.to_numpy()
        state_strings = np.array(['_'.join(map(str, state)) for state in state_tuples])
        state_ids, unique_states = pd.factorize(state_strings)
        state_mapping = {state: idx for idx, state in enumerate(unique_states)}
        num_states = len(state_mapping)

        state_frequencies = np.bincount(state_ids) / len(state_ids)

        print(f"Number of unique states: {num_states}")

        print("Step 3: Building smoothed transition matrix...")
        transition_matrix = self.build_smoothed_transition_matrix(state_ids, num_states)

        print("Step 4: Generating diverse sequences...")
        num_sequences = max(20, num_samples // 100)
        steps_per_sequence = num_samples // num_sequences

        generated_sequence = self.generate_diverse_sequences(
            transition_matrix, num_sequences, steps_per_sequence, state_frequencies
        )

        print("Step 5: Mapping to continuous values...")
        reverse_mapping = {v: k for k, v in state_mapping.items()}
        continuous_data = []

        for state_id in generated_sequence[:num_samples]:
            state_tuple = reverse_mapping[state_id]
            continuous_values = self.map_to_continuous_with_kde(state_tuple, bucket_edges)
            continuous_data.append(continuous_values)

        generated_df = pd.DataFrame(continuous_data, columns=data.columns)

        if add_noise:
            print("Step 6: Adding adaptive noise...")
            generated_df = self.add_gaussian_noise(generated_df)

        return generated_df


class CopulaMarkovGenerator:

    def __init__(self, markov_buckets=10, smoothing_alpha=0.01,
                 adaptive_bucketing=False, max_state_ratio=0.3,
                 max_buckets_per_feature=1000):
        self.copula_model = GaussianMultivariate()
        self.markov_gen = ImprovedMarkovGenerator(
            num_buckets=markov_buckets,
            smoothing_alpha=smoothing_alpha,
            use_kde=True,
            adaptive_bucketing=adaptive_bucketing,
            max_state_ratio=max_state_ratio,
            max_buckets_per_feature=max_buckets_per_feature,
        )

    def generate(self, data, num_samples=2000, mix_ratio=0.5, ensemble_mode='mixed', batch_size=100):
        if ensemble_mode == 'mixed':
            copula_samples = int(num_samples * mix_ratio)
            markov_samples = num_samples - copula_samples

            print(f"Generating {copula_samples} samples from Copula model...")
            try:
                self.copula_model.fit(data)
                copula_data = self.copula_model.sample(copula_samples)
                copula_data.columns = data.columns
            except Exception as e:
                print(f"Copula generation failed: {e}, using all Markov samples")
                copula_data = pd.DataFrame()
                markov_samples = num_samples

            print(f"Generating {markov_samples} samples from Markov chain...")
            markov_data = self.markov_gen.generate(data, markov_samples, add_noise=True)

            if not copula_data.empty:
                combined_data = pd.concat([copula_data, markov_data], ignore_index=True)
            else:
                combined_data = markov_data

        elif ensemble_mode == 'sequential':
            print(f"Generating samples sequentially with batch size: {batch_size}...")

            all_samples = []
            num_batches = num_samples // batch_size

            try:
                self.copula_model.fit(data)
                copula_available = True
            except:
                copula_available = False
                print("Copula not available, using only Markov")

            for i in range(num_batches):
                if i % 2 == 0 and copula_available:
                    batch = self.copula_model.sample(batch_size)
                    batch.columns = data.columns
                    print(f"Batch {i+1}/{num_batches}: Generated {batch_size} samples from Copula")
                else:
                    batch = self.markov_gen.generate(data, batch_size, add_noise=True)
                    print(f"Batch {i+1}/{num_batches}: Generated {batch_size} samples from Markov")

                all_samples.append(batch)

            remaining = num_samples - (num_batches * batch_size)
            if remaining > 0:
                if copula_available:
                    batch = self.copula_model.sample(remaining)
                    batch.columns = data.columns
                    print(f"Final batch: Generated {remaining} samples from Copula")
                else:
                    batch = self.markov_gen.generate(data, remaining, add_noise=True)
                    print(f"Final batch: Generated {remaining} samples from Markov")
                all_samples.append(batch)

            combined_data = pd.concat(all_samples, ignore_index=True)

        else:
            raise ValueError(f"Unknown ensemble_mode: {ensemble_mode}. Please use 'mixed' or 'sequential'.")

        combined_data = combined_data.sample(frac=1).reset_index(drop=True)

        print(f"Generated {len(combined_data)} samples using {ensemble_mode} mode")
        return combined_data


def main_copula_focused():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    start_row = 0
    end_row = 160

    file_path = os.path.join(SCRIPT_DIR, "data", "effluent_data_KT_AnMBR_Bac.csv")
    num_samples = 2000

    mix_ratio = 0.25
    ensemble_mode = 'sequential'
    batch_size = 100
    markov_buckets = 15
    smoothing_alpha = 0.01

    print(f"Configuration:")
    print(f"  - mix_ratio: {mix_ratio}")
    print(f"  - ensemble_mode: {ensemble_mode}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - markov_buckets: {markov_buckets}")
    print(f"  - smoothing_alpha: {smoothing_alpha}")
    print(f"  - num_samples: {num_samples}")

    print("\nLoading data...")
    loader = ImprovedMarkovGenerator(num_buckets=5)
    data = loader.load_data(file_path, start_row, end_row)
    print(f"Data shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")

    print("\n" + "="*50)
    print("Copula-Markov Hybrid Generation (CM)")
    print("="*50)

    cm_gen = CopulaMarkovGenerator(
        markov_buckets=markov_buckets,
        smoothing_alpha=smoothing_alpha
    )

    print(f"\nGenerating synthetic data with Copula-Markov hybrid...")
    generated_data = cm_gen.generate(
        data,
        num_samples=num_samples,
        mix_ratio=mix_ratio,
        ensemble_mode=ensemble_mode,
        batch_size=batch_size
    )

    print(f"\nGenerated data shape: {generated_data.shape}")
    print("Generated data statistics:")
    print(generated_data.describe())

    output_path = os.path.join(SCRIPT_DIR, "output", "emcm-gcrw-effluent_data_KT_AnMBR_Bac.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generated_data.to_csv(output_path, index=False)
    print(f"\nGenerated data saved to: {output_path}")

    return generated_data


if __name__ == "__main__":
    main_copula_focused()
