const Config = {

    // ── Backend ─────────────────────────────────────────────────────────────
    api: {
        baseUrl: '',                 // '' = same origin (Flask serves the UI)
        endpoints: {
            emcm_ps:          '/api/generate/emcm-ps',
            lstmTrain:        '/api/lstm/train',
            lstmValidate:     '/api/lstm/validate',
            lstmPred:         '/api/lstm/predict',
            lifelongOptimize: '/api/lifelong/optimize',
            lifelongTrain:    '/api/lifelong/train',
            lifelongAdapt:    '/api/lifelong/adapt',
        },
    },

    // ── Sides ───────────────────────────────────────────────────────────────
    // Both sides run the same algorithm; the second panel is only shown when
    // the "Two panels" view mode is selected.
    sides: {
        left:  { algorithm: 'emcm_ps', title: 'EMCM-PS' },
        right: { algorithm: 'emcm_ps', title: 'EMCM-PS' },
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
            description: 'Markov Chain synthetic generator with correlation-aware augmentation and local-distribution decoding.',
            params: [
                { key: 'num_steps',             label: 'Synthetic rows',            type: 'int',    min: 50,   max: 20000, step: 100,    default: 2000,  hint: 'How many synthetic rows to generate' },
                { key: 'num_augmented_samples', label: 'Augmented rows (noise)',    type: 'int',    min: 0,    max: 5000,  step: 50,     default: 300,   hint: 'Correlation-preserving rows added before discretization' },
                { key: 'noise_level',           label: 'Noise level',               type: 'float',  min: 0,    max: 0.5,   step: 0.005,  default: 0.02,  hint: 'Multivariate noise std used during augmentation' },
                { key: 'num_buckets',           label: 'Num buckets',               type: 'int',    min: 2,    max: 30,    step: 1,      default: 15,     hint: 'Fixed quantile buckets per column' },
                { key: 'decoder_mode',          label: 'Decoder mode',              type: 'select', options: ['local','uniform'],        default: 'local', hint: 'local = multivariate-normal / convex-combo \n uniform = uniform inside bucket' },
            ],
        },
    },
};
