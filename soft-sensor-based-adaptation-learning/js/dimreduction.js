const DimReduction = {

    MAX_POINTS: 600,  // subsample synthetic if combined exceeds this

    // ── Seeded PRNG (reproducible results) ─────────────────────────────────────

    _seededRng(seed) {
        let s = seed;
        return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    },

    // ── Matrix helpers ─────────────────────────────────────────────────────────

    /** Rows-of-numbers matrix from row-objects */
    _toMatrix(data, columns) {
        return data.map(row => columns.map(c => {
            const v = typeof row[c] === 'string' ? parseFloat(row[c]) : Number(row[c]);
            return isFinite(v) ? v : 0;
        }));
    },

    /** Centre + scale each column (returns { data, means, stds }) */
    _standardise(matrix) {
        const n = matrix.length, d = matrix[0].length;
        const means = new Array(d).fill(0);
        const stds  = new Array(d).fill(0);

        for (let i = 0; i < n; i++)
            for (let j = 0; j < d; j++)
                means[j] += matrix[i][j];
        for (let j = 0; j < d; j++) means[j] /= n;

        for (let i = 0; i < n; i++)
            for (let j = 0; j < d; j++)
                stds[j] += (matrix[i][j] - means[j]) ** 2;
        for (let j = 0; j < d; j++) stds[j] = Math.sqrt(stds[j] / Math.max(n - 1, 1)) || 1;

        const data = matrix.map(row => row.map((v, j) => (v - means[j]) / stds[j]));
        return { data, means, stds };
    },

    /** Covariance matrix of (already centred) data */
    _covMatrix(data) {
        const n = data.length, d = data[0].length;
        const cov = Array.from({ length: d }, () => new Float64Array(d));
        for (let i = 0; i < d; i++)
            for (let j = i; j < d; j++) {
                let s = 0;
                for (let k = 0; k < n; k++) s += data[k][i] * data[k][j];
                cov[i][j] = cov[j][i] = s / Math.max(n - 1, 1);
            }
        return cov;
    },

    /** Power iteration → { vector, value } */
    _powerIter(matrix, rng, nIter = 300) {
        const d = matrix.length;
        let v = Array.from({ length: d }, () => rng() - 0.5);
        for (let iter = 0; iter < nIter; iter++) {
            const nv = new Array(d).fill(0);
            for (let i = 0; i < d; i++)
                for (let j = 0; j < d; j++)
                    nv[i] += matrix[i][j] * v[j];
            const norm = Math.sqrt(nv.reduce((a, x) => a + x * x, 0)) || 1;
            v = nv.map(x => x / norm);
        }
        // eigenvalue
        const mv = new Array(d).fill(0);
        for (let i = 0; i < d; i++)
            for (let j = 0; j < d; j++)
                mv[i] += matrix[i][j] * v[j];
        const value = v.reduce((a, x, i) => a + x * mv[i], 0);
        return { vector: v, value };
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // PCA
    // ═══════════════════════════════════════════════════════════════════════════

    /** Project real + synthetic into 2-D PCA space (standardisation fitted on the combined dataset, matches sklearn) */
    pca(realData, synthData, columns) {
        const rng = DimReduction._seededRng(42);
        const rMatrix = DimReduction._toMatrix(realData, columns);
        let   sMatrix = DimReduction._toMatrix(synthData, columns);

        // Subsample synthetic if too large
        const maxS = DimReduction.MAX_POINTS - rMatrix.length;
        if (sMatrix.length > maxS) sMatrix = DimReduction._sample(sMatrix, maxS, rng);

        const combined = [...rMatrix, ...sMatrix];
        const { data: std } = DimReduction._standardise(combined);
        const cov = DimReduction._covMatrix(std);

        // 1st component
        const pc1 = DimReduction._powerIter(cov, rng);
        // Deflate
        const d = cov.length;
        const deflated = cov.map((row, i) =>
            Array.from(row, (val, j) => val - pc1.value * pc1.vector[i] * pc1.vector[j])
        );
        // 2nd component
        const pc2 = DimReduction._powerIter(deflated, rng);

        // Project
        const project = (row) => ({
            x: row.reduce((a, v, i) => a + v * pc1.vector[i], 0),
            y: row.reduce((a, v, i) => a + v * pc2.vector[i], 0),
        });

        const nR = rMatrix.length;
        return {
            real:  std.slice(0, nR).map(project),
            synth: std.slice(nR).map(project),
        };
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // t-SNE (basic O(n²) — perfectly fine for n ≤ 600)
    // ═══════════════════════════════════════════════════════════════════════════

    tsne(realData, synthData, columns, opts = {}) {
        const { perplexity: perpIn, maxIter = 300, lr = 150 } = opts;
        const rng = DimReduction._seededRng(42);

        const rMatrix = DimReduction._toMatrix(realData, columns);
        let   sMatrix = DimReduction._toMatrix(synthData, columns);
        const maxS = DimReduction.MAX_POINTS - rMatrix.length;
        if (sMatrix.length > maxS) sMatrix = DimReduction._sample(sMatrix, maxS, rng);

        const combined = [...rMatrix, ...sMatrix];
        const n = combined.length, d = combined[0].length;
        if (n < 4) return { real: [], synth: [] };

        const { data: X } = DimReduction._standardise(combined);

        // Pairwise squared distances
        const D2 = Array.from({ length: n }, () => new Float64Array(n));
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                let s = 0;
                for (let k = 0; k < d; k++) s += (X[i][k] - X[j][k]) ** 2;
                D2[i][j] = D2[j][i] = s;
            }
        }

        // Perplexity (auto-clamp)
        const perp = perpIn || Math.min(30, Math.max(2, Math.floor(n / 4)));
        const targetH = Math.log(perp);

        // ── Compute P ──
        const P = Array.from({ length: n }, () => new Float64Array(n));
        for (let i = 0; i < n; i++) {
            let lo = 1e-10, hi = 1e4, sigma = 1;
            for (let iter = 0; iter < 50; iter++) {
                sigma = (lo + hi) / 2;
                const twoS2 = 2 * sigma * sigma;
                let sumExp = 0;
                for (let j = 0; j < n; j++) { if (j !== i) sumExp += Math.exp(-D2[i][j] / twoS2); }
                let H = 0;
                for (let j = 0; j < n; j++) {
                    if (j === i) continue;
                    const p = Math.exp(-D2[i][j] / twoS2) / (sumExp || 1e-10);
                    if (p > 1e-12) H -= p * Math.log(p);
                }
                if (H > targetH) hi = sigma; else lo = sigma;
            }
            const twoS2 = 2 * sigma * sigma;
            let sumExp = 0;
            for (let j = 0; j < n; j++) { if (j !== i) sumExp += Math.exp(-D2[i][j] / twoS2); }
            for (let j = 0; j < n; j++) {
                if (j === i) continue;
                P[i][j] = Math.exp(-D2[i][j] / twoS2) / (sumExp || 1e-10);
            }
        }
        // Symmetrise
        for (let i = 0; i < n; i++)
            for (let j = i + 1; j < n; j++) {
                const val = Math.max((P[i][j] + P[j][i]) / (2 * n), 1e-12);
                P[i][j] = P[j][i] = val;
            }

        // ── Gradient descent ──
        let Y = Array.from({ length: n }, () => [(rng() - 0.5) * 1e-4, (rng() - 0.5) * 1e-4]);
        const gains  = Array.from({ length: n }, () => [1, 1]);
        const update = Array.from({ length: n }, () => [0, 0]);

        for (let t = 0; t < maxIter; t++) {
            const exag = t < 100 ? 4 : 1;
            // Q distribution
            let sumQ = 0;
            const Qnum = Array.from({ length: n }, () => new Float64Array(n));
            for (let i = 0; i < n; i++)
                for (let j = i + 1; j < n; j++) {
                    const d2 = (Y[i][0] - Y[j][0]) ** 2 + (Y[i][1] - Y[j][1]) ** 2;
                    const q = 1 / (1 + d2);
                    Qnum[i][j] = Qnum[j][i] = q;
                    sumQ += 2 * q;
                }

            // Gradients
            const grad = Array.from({ length: n }, () => [0, 0]);
            for (let i = 0; i < n; i++)
                for (let j = 0; j < n; j++) {
                    if (i === j) continue;
                    const q = Qnum[i][j] / (sumQ || 1e-10);
                    const mult = 4 * (exag * P[i][j] - q) * Qnum[i][j];
                    grad[i][0] += mult * (Y[i][0] - Y[j][0]);
                    grad[i][1] += mult * (Y[i][1] - Y[j][1]);
                }

            const mom = t < 250 ? 0.5 : 0.8;
            for (let i = 0; i < n; i++)
                for (let dim = 0; dim < 2; dim++) {
                    gains[i][dim] = (Math.sign(grad[i][dim]) !== Math.sign(update[i][dim]))
                        ? gains[i][dim] + 0.2 : Math.max(gains[i][dim] * 0.8, 0.01);
                    update[i][dim] = mom * update[i][dim] - lr * gains[i][dim] * grad[i][dim];
                    Y[i][dim] += update[i][dim];
                }
            // Centre
            let cy0 = 0, cy1 = 0;
            for (let i = 0; i < n; i++) { cy0 += Y[i][0]; cy1 += Y[i][1]; }
            cy0 /= n; cy1 /= n;
            for (let i = 0; i < n; i++) { Y[i][0] -= cy0; Y[i][1] -= cy1; }
        }

        const nR = rMatrix.length;
        return {
            real:  Y.slice(0, nR).map(p => ({ x: p[0], y: p[1] })),
            synth: Y.slice(nR).map(p    => ({ x: p[0], y: p[1] })),
        };
    },

    // ── Utility ────────────────────────────────────────────────────────────────

    /** Random sample of rows from a matrix */
    _sample(matrix, k, rng) {
        const indices = Array.from({ length: matrix.length }, (_, i) => i);
        // Fisher-Yates partial shuffle
        for (let i = indices.length - 1; i > 0 && i >= indices.length - k; i--) {
            const j = Math.floor(rng() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        return indices.slice(-k).map(i => matrix[i]);
    },
};
