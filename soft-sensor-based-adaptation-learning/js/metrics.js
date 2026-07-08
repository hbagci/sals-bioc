const Metrics = {

    // ═══════════════════════════════════════════════════════════════════════════
    // PER-COLUMN METRICS
    // ═══════════════════════════════════════════════════════════════════════════

    /** KS Complement — 1 − max|CDF_real − CDF_synth| (SDMetrics KSComplement) */
    ksComplement(realArr, synthArr) {
        const rSorted = Stats.clean(realArr).sort((a, b) => a - b);
        const sSorted = Stats.clean(synthArr).sort((a, b) => a - b);
        const nR = rSorted.length, nS = sSorted.length;
        if (nR === 0 || nS === 0) return NaN;
        let i = 0, j = 0, maxDiff = 0;
        while (i < nR || j < nS) {
            const rVal = i < nR ? rSorted[i] : Infinity;
            const sVal = j < nS ? sSorted[j] : Infinity;
            if (rVal <= sVal) i++;
            if (sVal <= rVal) j++;
            maxDiff = Math.max(maxDiff, Math.abs(i / nR - j / nS));
        }
        return 1 - maxDiff;
    },

    /** Jensen-Shannon Divergence Score — 1 − JSD(P, Q), log base 2 so JSD ∈ [0,1] */
    jsdScore(realArr, synthArr, numBins) {
        const rClean = Stats.clean(realArr), sClean = Stats.clean(synthArr);
        if (rClean.length === 0 || sClean.length === 0) return NaN;
        const allMin = Math.min(Stats.min(rClean), Stats.min(sClean));
        const allMax = Math.max(Stats.max(rClean), Stats.max(sClean));
        if (allMin === allMax) return 1;
        const bins  = numBins || Math.max(5, Math.ceil(1 + 3.322 * Math.log10(rClean.length + sClean.length)));
        const width = (allMax - allMin) / bins;
        const binCount = arr => {
            const h = new Float64Array(bins);
            arr.forEach(v => { h[Math.min(bins - 1, Math.floor((v - allMin) / width))]++; });
            return Array.from(h, c => (c / arr.length) || 1e-12);
        };
        const P = binCount(rClean), Q = binCount(sClean);
        const M = P.map((p, k) => (p + Q[k]) / 2);
        const kl = (a, b) => a.reduce((s, ai, k) => s + (ai > 1e-12 ? ai * Math.log2(ai / b[k]) : 0), 0);
        return Math.max(0, 1 - (kl(P, M) + kl(Q, M)) / 2);
    },

    /** Wasserstein-1 Distance (normalised) — 1 − W₁/range */
    wasserstein1DScore(realArr, synthArr) {
        const rClean = Stats.clean(realArr), sClean = Stats.clean(synthArr);
        if (rClean.length === 0 || sClean.length === 0) return NaN;
        const range = Stats.max(rClean) - Stats.min(rClean);
        if (range === 0) return sClean.every(v => v === rClean[0]) ? 1 : 0;
        const rSorted = [...rClean].sort((a, b) => a - b);
        const sSorted = [...sClean].sort((a, b) => a - b);
        const steps = 1000;
        let sum = 0;
        for (let k = 0; k < steps; k++) {
            const t = (k + 0.5) / steps;
            sum += Math.abs(Stats.quantileFromSorted(rSorted, t) - Stats.quantileFromSorted(sSorted, t));
        }
        return Math.max(0, 1 - (sum / steps) / range);
    },

    /** Range Coverage — SDMetrics definition */
    rangeCoverage(realArr, synthArr) {
        const rClean = Stats.clean(realArr), sClean = Stats.clean(synthArr);
        if (rClean.length === 0 || sClean.length === 0) return NaN;
        const rMin = Stats.min(rClean), rMax = Stats.max(rClean);
        const range = rMax - rMin;
        if (range === 0) return (Stats.min(sClean) <= rMin && Stats.max(sClean) >= rMax) ? 1 : 0;
        return Math.max(0, 1 - Math.max(0, Stats.min(sClean) - rMin) / range
                            - Math.max(0, rMax - Stats.max(sClean)) / range);
    },

    /** Boundary Adherence — % of synthetic within [min_real ±10%, max_real ±10%] */
    boundaryAdherence(realArr, synthArr) {
        const rClean = Stats.clean(realArr), sClean = Stats.clean(synthArr);
        if (rClean.length === 0 || sClean.length === 0) return NaN;
        const tol = (Stats.max(rClean) - Stats.min(rClean)) * 0.10;
        const lo = Stats.min(rClean) - tol, hi = Stats.max(rClean) + tol;
        return sClean.filter(v => v >= lo && v <= hi).length / sClean.length;
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // GLOBAL METRICS
    // ═══════════════════════════════════════════════════════════════════════════

    /** Correlation Similarity — SDMetrics: avg of per-pair 1 − |Δr|/2 */
    correlationSimilarity(realData, synthData, columns) {
        const corrR = Stats.correlationMatrix(realData, columns);
        const corrS = Stats.correlationMatrix(synthData, columns);
        let sum = 0, count = 0;
        for (let i = 0; i < columns.length; i++)
            for (let j = i + 1; j < columns.length; j++) {
                sum += 1 - Math.abs((isNaN(corrR[i][j]) ? 0 : corrR[i][j])
                                  - (isNaN(corrS[i][j]) ? 0 : corrS[i][j])) / 2;
                count++;
            }
        return count > 0 ? sum / count : 1;
    },

    /** Correlation Frobenius Score — normalised RMS of the off-diagonal correlation difference matrix (1 = perfect, 0 = max difference) */
    corrFrobeniusScore(realData, synthData, columns) {
        const corrR = Stats.correlationMatrix(realData, columns);
        const corrS = Stats.correlationMatrix(synthData, columns);
        let sumSq = 0, count = 0;
        for (let i = 0; i < columns.length; i++)
            for (let j = i + 1; j < columns.length; j++) {
                const d = (corrR[i][j] || 0) - (corrS[i][j] || 0);
                sumSq += d * d;
                count++;
            }
        if (count === 0) return 1;
        // Normalise: max possible RMS = 2 (corr swings from -1 to +1)
        return Math.max(0, 1 - Math.sqrt(sumSq / count) / 2);
    },

    /** PCA Alignment Score — overlap of real/synthetic clouds in PCA space (centroid distance + spread similarity); returns { score, projection } reused by charts.js */
    pcaMetrics(realData, synthData, columns) {
        if (columns.length < 2) return { score: NaN, projection: { real: [], synth: [] } };
        const projection = DimReduction.pca(realData, synthData, columns);
        const { real, synth } = projection;
        if (real.length === 0 || synth.length === 0) return { score: NaN, projection };

        const rX = real.map(p => p.x),  rY = real.map(p => p.y);
        const sX = synth.map(p => p.x), sY = synth.map(p => p.y);

        const cRx = Stats.mean(rX), cRy = Stats.mean(rY);
        const cSx = Stats.mean(sX), cSy = Stats.mean(sY);
        const spreadRx = Stats.std(rX) || 1, spreadRy = Stats.std(rY) || 1;

        // Normalised centroid distance (divide by real spread so units cancel)
        const centroidDist = Math.sqrt(
            ((cRx - cSx) / spreadRx) ** 2 + ((cRy - cSy) / spreadRy) ** 2
        );
        const centroidScore = Math.max(0, 1 - centroidDist / Math.sqrt(2));

        // Spread ratio in each PC direction
        const spreadSx = Stats.std(sX) || 0, spreadSy = Stats.std(sY) || 0;
        const spreadScore = Math.max(0, 1 - (
            Math.abs(spreadRx - spreadSx) / spreadRx +
            Math.abs(spreadRy - spreadSy) / spreadRy
        ) / 2);

        return { score: (centroidScore + spreadScore) / 2, projection };
    },

    /** t-SNE Coverage Score — avg distance from each real point to its nearest synthetic neighbour in t-SNE space; returns { score, projection } reused by charts.js */
    tsneMetrics(realData, synthData, columns) {
        if (columns.length < 2) return { score: NaN, projection: { real: [], synth: [] } };
        const projection = DimReduction.tsne(realData, synthData, columns);
        const { real, synth } = projection;
        if (real.length === 0 || synth.length === 0) return { score: NaN, projection };

        // Distance from each real point to nearest synthetic point
        const nearestDists = real.map(r => {
            let minD = Infinity;
            synth.forEach(s => {
                const d = Math.sqrt((r.x - s.x) ** 2 + (r.y - s.y) ** 2);
                if (d < minD) minD = d;
            });
            return minD;
        });

        // Normalise by the overall spread of the combined embedding
        const allX = [...real.map(p => p.x), ...synth.map(p => p.x)];
        const spread = Stats.std(allX) || 1;
        const avgDist = Stats.mean(nearestDists) || 0;
        const score = Math.max(0, 1 - avgDist / spread);

        return { score, projection };
    },

    /** DCR Baseline Protection — privacy diagnostic (not included in overall score) */
    dcrPrivacy(realData, synthData, columns) {
        const realMatrix  = columns.map(c => Stats.getValues(realData, c));
        const synthMatrix = columns.map(c => Stats.getValues(synthData, c));
        const nR = realMatrix[0].length, nS = synthMatrix[0].length, nC = columns.length;
        if (nR === 0 || nS === 0) return NaN;
        const ranges = realMatrix.map(col => { const r = Stats.max(col) - Stats.min(col); return r === 0 ? 1 : r; });
        const medianDCR = (queryMatrix, refMatrix) => {
            const nQ = queryMatrix[0].length, nRef = refMatrix[0].length;
            const step = Math.max(1, Math.floor(nQ / 200));
            const dists = [];
            for (let qi = 0; qi < nQ; qi += step) {
                let minD = Infinity;
                for (let ri = 0; ri < nRef; ri++) {
                    let d = 0;
                    for (let c = 0; c < nC; c++) d += Math.abs(queryMatrix[c][qi] - refMatrix[c][ri]) / ranges[c];
                    d /= nC;
                    if (d < minD) minD = d;
                }
                dists.push(minD);
            }
            dists.sort((a, b) => a - b);
            return Stats.quantileFromSorted(dists, 0.5);
        };
        const synthDCR = medianDCR(synthMatrix, realMatrix);
        const mins = realMatrix.map(col => Stats.min(col));
        const maxs = realMatrix.map(col => Stats.max(col));
        const nRand = Math.min(nS, 500);
        const randMatrix = columns.map((_, c) => {
            const arr = [];
            for (let i = 0; i < nRand; i++) arr.push(mins[c] + Math.random() * (maxs[c] - mins[c]));
            return arr;
        });
        const randDCR = medianDCR(randMatrix, realMatrix);
        return randDCR > 0 ? Math.min(1, synthDCR / randDCR) : 1;
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // AGGREGATE REPORT  (async — runs PCA + t-SNE once, returns projections too)
    // ═══════════════════════════════════════════════════════════════════════════

    /** Compute the full quality report (async so panel.js can show a "Computing…" indicator); returns pre-computed PCA/t-SNE projections too so charts.js doesn't re-run them */
    async computeReport(realData, synthData, columns) {
        // --- Per-column ---
        const columnMetrics = columns.map(col => {
            const rVals = Stats.getValues(realData, col);
            const sVals = Stats.getValues(synthData, col);
            return {
                column:      col,
                ks:          Metrics.ksComplement(rVals, sVals),
                jsd:         Metrics.jsdScore(rVals, sVals),
                wasserstein: Metrics.wasserstein1DScore(rVals, sVals),
                rangeCov:    Metrics.rangeCoverage(rVals, sVals),
                boundary:    Metrics.boundaryAdherence(rVals, sVals),
            };
        });

        // --- Global distribution metrics ---
        const columnShapes    = Metrics._avg(columnMetrics.map(m => m.ks));
        const pairTrends      = Metrics.correlationSimilarity(realData, synthData, columns);
        const corrFrob        = Metrics.corrFrobeniusScore(realData, synthData, columns);
        const avgJSD          = Metrics._avg(columnMetrics.map(m => m.jsd));
        const avgWasserstein  = Metrics._avg(columnMetrics.map(m => m.wasserstein));
        const avgRangeCov     = Metrics._avg(columnMetrics.map(m => m.rangeCov));
        const avgBoundary     = Metrics._avg(columnMetrics.map(m => m.boundary));

        // --- PCA (yield to browser between heavy steps) ---
        await new Promise(r => setTimeout(r, 0));
        const { score: pcaScore, projection: pcaProjection } = Metrics.pcaMetrics(realData, synthData, columns);

        // --- t-SNE ---
        await new Promise(r => setTimeout(r, 0));
        const { score: tsneScore, projection: tsneProjection } = Metrics.tsneMetrics(realData, synthData, columns);

        // --- Privacy (diagnostic only — NOT in overall score) ---
        const privacy = Metrics.dcrPrivacy(realData, synthData, columns);

        // --- Overall Quality Score (weights from Config, auto-normalised) ---
        // Disabled metrics (Config.metrics[k] === false) are excluded from the score.
        const w = Config.weights;
        const enabled = Config.metrics || {};
        const ew = k => (enabled[k] === false ? 0 : (w[k] || 0));
        const totalW = Object.keys(w).reduce((s, k) => s + Math.max(0, ew(k)), 0) || 1;
        const overall = (
            ew('columnShapes')   * columnShapes   +
            ew('pairTrends')     * pairTrends     +
            ew('corrFrob')       * corrFrob       +
            ew('avgJSD')         * avgJSD         +
            ew('avgWasserstein') * avgWasserstein +
            ew('pcaScore')       * pcaScore       +
            ew('tsneScore')      * tsneScore      +
            ew('avgRangeCov')    * avgRangeCov    +
            ew('avgBoundary')    * avgBoundary
        ) / totalW;

        return {
            columnMetrics,
            // Scorecard bars
            columnShapes, pairTrends, corrFrob,
            avgJSD, avgWasserstein, avgRangeCov, avgBoundary,
            pcaScore, tsneScore,
            // Diagnostic
            privacy,
            // Overall
            overall,
            // Pre-computed projections for charts (avoid re-running)
            pcaProjection,
            tsneProjection,
        };
    },

    regression(yTrue, yPred) {
        const n = yTrue.length;
        if (n === 0) return { R2: NaN, RMSE: NaN, MAE: NaN, MSE: NaN };
        const mean  = yTrue.reduce((s, v) => s + v, 0) / n;
        const ssTot = yTrue.reduce((s, v) => s + (v - mean) ** 2, 0);
        const ssRes = yTrue.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0);
        const MSE   = ssRes / n;
        const RMSE  = Math.sqrt(MSE);
        const MAE   = yTrue.reduce((s, v, i) => s + Math.abs(v - yPred[i]), 0) / n;
        const R2    = ssTot > 0 ? 1 - ssRes / ssTot : NaN;
        return { R2, RMSE, MAE, MSE };
    },

    /** Safe average (ignores NaN) */
    _avg(arr) {
        const valid = arr.filter(v => isFinite(v));
        return valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : NaN;
    },
};
