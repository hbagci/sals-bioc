const Config = {

    // ── Backend ─────────────────────────────────────────────────────────────
    api: {
        baseUrl: '',                 // '' = same origin (Flask serves the UI)
        endpoints: {
            emcm_ps:          '/api/generate/emcm-ps',
            emcm_gcrw:        '/api/generate/emcm-gcrw',
            lstmTrain:        '/api/lstm/train',
            lstmValidate:     '/api/lstm/validate',
            lstmPred:         '/api/lstm/predict',
            lifelongOptimize: '/api/lifelong/optimize',
            lifelongTrain:    '/api/lifelong/train',
            lifelongAdapt:    '/api/lifelong/adapt',
        },
    },

    // ── Sides ───────────────────────────────────────────────────────────────
    // Which algorithm runs on which side of the comparison view.
    sides: {
        left:  { algorithm: 'emcm_gcrw', title: 'EMCM-GCRW' },
        right: { algorithm: 'emcm_ps',   title: 'EMCM-PS' },
    },

    // ── Section visibility ──────────────────────────────────────────────────
    // Toggle whole UI sections on / off. Applies to BOTH sides.
    show: {
        // Top-of-panel controls
        paramsPanel:         true,   // Parameter editor (per-algorithm)
        autoRegenerate:      true,   // "Auto-regenerate on change" toggle
        downloadButton:      true,   // Download generated CSV button

        // Quality sections
        scorecard:           true,   // Quality Assessment ring + breakdown bars
        statsComparison:     true,   // Mean / Std / Skew / Kurtosis comparison table
        overlaidHistograms:  true,   // Real vs synthetic histograms per column
        boxPlots:            true,   // Real vs synthetic box plots per column
        correlationHeatmaps: true,   // Real / Synth / Diff correlation matrices
        pcaScatter:          true,   // PCA projection scatter
        tsneScatter:         true,   // t-SNE projection scatter
        metricsDetail:       false,   // Per-column detailed metrics table
        bucketingIterations: true,   // EMCM-PS adaptive-bucketing iteration trace (right side only; left stays blank)
        lifelongLSTM:        true,   // Lifelong-LSTM section below LSTM section
    },

    // ── Per-metric enable toggles ───────────────────────────────────────────
    // Turn individual Quality Assessment metrics on / off.
    // A disabled metric is hidden from the breakdown bars AND excluded from
    // the Overall Quality Score (its weight is treated as 0).
    metrics: {
        columnShapes:   true,   // Column Shapes (KS)
        avgJSD:         false,   // Distribution (JSD)
        avgWasserstein: false,   // Earth Mover (W₁)
        avgRangeCov:    true,   // Range Coverage
        avgBoundary:    false,   // Boundary Adherence
        pairTrends:     true,   // Pair Trends (avg correlation)
        corrFrob:       true,   // Correlation Frobenius RMS
        pcaScore:       true,   // PCA Alignment
        tsneScore:      true,   // t-SNE Coverage
        privacy:        false,   // Privacy (DCR) — diagnostic only
    },

    // ── Metric weights for Overall Quality Score ────────────────────────────
    // Any positive number is valid — normalised to sum to 1.0 at runtime.
    // Set a weight to 0 to exclude that metric from the overall score.
    weights: {
        columnShapes:   0.05,   // KS Complement
        avgJSD:         0.08,   // Jensen-Shannon
        avgWasserstein: 0.07,   // Wasserstein-1
        avgRangeCov:    0.05,   // Range Coverage
        avgBoundary:    0.05,   // Boundary Adherence
        pairTrends:     0.30,   // Correlation Avg (pairwise trends)
        corrFrob:       0.30,   // Frobenius RMS
        pcaScore:       0.15,   // PCA Alignment
        tsneScore:      0.15,   // t-SNE Coverage
    },

    // ── Per-algorithm parameter schemas ─────────────────────────────────────
    // Each entry drives the parameter editor UI on its side.
    //
    // Field types:
    //   int    — number input, integer step
    //   float  — number input, decimal step
    //   bool   — checkbox
    //   select — dropdown from `options`
    //
    // Optional keys:
    //   min, max, step  — numeric bounds
    //   hint            — short help text shown under the input
    //   visible         — false hides the field from the UI (still sent with default)
    //   showIf          — (params) => boolean; hide field unless predicate true
    //
    algorithms: {
        emcm_ps: {
            label: 'EMCM-PS — emcm_ps.py',
            description: 'Markov Chain synthetic generator with correlation-aware augmentation, adaptive bucketing, and local-distribution decoding.',
            params: [
                { key: 'num_steps',             label: 'Synthetic rows',            type: 'int',    min: 50,   max: 20000, step: 100,    default: 2000,  hint: 'How many synthetic rows to generate' },
                { key: 'num_augmented_samples', label: 'Augmented rows (noise)',    type: 'int',    min: 0,    max: 5000,  step: 50,     default: 300,   hint: 'Correlation-preserving rows added before discretization' },
                { key: 'noise_level',           label: 'Noise level',               type: 'float',  min: 0,    max: 0.5,   step: 0.005,  default: 0.02,  hint: 'Multivariate noise std used during augmentation' },
                { key: 'adaptive_bucketing',    label: 'Adaptive bucketing',        type: 'bool',                                        default: false,  hint: 'True = recursive largest-gap splitting (ghost buckets for empty regions); False = fixed quantile buckets' },
                { key: 'max_state_ratio',       label: 'Max state ratio',           type: 'float',  min: 0.05, max: 1.0,   step: 0.05,   default: 0.05,   hint: 'Splitting stops once joint-state ratio would exceed this', showIf: p => p.adaptive_bucketing },
                { key: 'max_buckets_per_feature', label: 'Max buckets / feature',   type: 'int',    min: 2,    max: 10000, step: 1,      default: 1000,  hint: 'Per-column cap on bucket count (adaptive only)', showIf: p => p.adaptive_bucketing },
                { key: 'num_buckets',           label: 'Num buckets (fixed)',       type: 'int',    min: 2,    max: 30,    step: 1,      default: 15,     hint: 'Used when adaptive bucketing is off',       showIf: p => !p.adaptive_bucketing },
                { key: 'decoder_mode',          label: 'Decoder mode',              type: 'select', options: ['local','uniform'],        default: 'local', hint: 'local = multivariate-normal / convex-combo \n uniform = uniform inside bucket' },
            ],
        },

        emcm_gcrw: {
            label: 'EMCM-GCRW — emcm_gcrw.py',
            description: 'Ensemble-SDA: Copula-Markov hybrid. Combines a Gaussian copula with an improved Markov chain using KDE-based decoding and Laplace smoothing.',
            params: [
                { key: 'num_samples',           label: 'Synthetic rows',          type: 'int',    min: 50,   max: 20000, step: 100,   default: 2000 },
                { key: 'adaptive_bucketing',    label: 'Adaptive bucketing',       type: 'bool',                                        default: false, hint: 'True = recursive largest-gap splitting (same as EMCM-PS); False = fixed quantile buckets' },
                { key: 'markov_buckets',        label: 'Markov buckets (fixed)',   type: 'int',    min: 3,    max: 50,    step: 1,     default: 15,  hint: 'Discretization granularity when adaptive bucketing is off', showIf: p => !p.adaptive_bucketing },
                { key: 'max_state_ratio',       label: 'Max state ratio',          type: 'float',  min: 0.05, max: 1.0,   step: 0.05,  default: 0.3,  hint: 'Splitting stops once joint-state ratio would exceed this', showIf: p => p.adaptive_bucketing },
                { key: 'max_buckets_per_feature', label: 'Max buckets / feature',  type: 'int',    min: 2,    max: 10000, step: 1,     default: 1000, hint: 'Per-column cap on bucket count (adaptive only)',          showIf: p => p.adaptive_bucketing },
                { key: 'smoothing_alpha',       label: 'Laplace smoothing α',      type: 'float',  min: 0,    max: 1,     step: 0.005, default: 0.01, hint: 'Prevents zero-probability transitions' },
                { key: 'mix_ratio',             label: 'Copula mix ratio',         type: 'float',  min: 0,    max: 1,     step: 0.05,  default: 0.25, hint: 'Fraction of samples from the Copula model' },
                { key: 'ensemble_mode',         label: 'Ensemble mode',            type: 'select', options: ['mixed','sequential'],    default: 'mixed', hint: 'mixed = concat once; sequential = alternate batches' },
                { key: 'batch_size',            label: 'Batch size',               type: 'int',    min: 10,   max: 2000,  step: 10,    default: 100,  hint: 'Batch size for sequential mode', showIf: p => p.ensemble_mode === 'sequential' },
                // Hidden — kept for parity with the original EMCM-GCRW main() which slices rows 0..160.
                { key: 'start_row',             label: 'Input start row',          type: 'int',    default: 0,   visible: false },
                { key: 'end_row',               label: 'Input end row',            type: 'int',    default: 160, visible: false },
            ],
        },
    },
};
