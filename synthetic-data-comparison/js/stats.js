const Stats = {

    // ── Array helpers ──────────────────────────────────────────────────────────

    /** Strip non-finite values, cast to float */
    clean(arr) {
        return arr.map(v => (typeof v === 'string' ? parseFloat(v) : Number(v)))
                  .filter(v => isFinite(v) && !isNaN(v));
    },

    /** Extract a column from row-object data as clean numbers */
    getValues(data, col) {
        return Stats.clean(data.map(row => row[col]));
    },

    // ── Central tendency & dispersion ──────────────────────────────────────────

    mean(arr) {
        const v = Stats.clean(arr);
        return v.length === 0 ? NaN : v.reduce((a, b) => a + b, 0) / v.length;
    },

    /** Sample variance (ddof=1, matches numpy) */
    variance(arr, mu) {
        const v = Stats.clean(arr);
        if (v.length < 2) return NaN;
        const m = mu !== undefined ? mu : Stats.mean(v);
        return v.reduce((acc, x) => acc + (x - m) ** 2, 0) / (v.length - 1);
    },

    std(arr, mu) {
        const va = Stats.variance(arr, mu);
        return isNaN(va) ? NaN : Math.sqrt(va);
    },

    // ── Quantiles ──────────────────────────────────────────────────────────────

    /** Linear-interpolation quantile (matches numpy default) */
    quantile(arr, q) {
        const v = Stats.clean(arr).sort((a, b) => a - b);
        return Stats.quantileFromSorted(v, q);
    },

    /** Quantile on a pre-sorted array (avoids re-sorting) */
    quantileFromSorted(sorted, q) {
        if (sorted.length === 0) return NaN;
        const pos = q * (sorted.length - 1);
        const lo = Math.floor(pos), hi = Math.ceil(pos);
        if (lo === hi) return sorted[lo];
        return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
    },

    median(arr)  { return Stats.quantile(arr, 0.5); },
    min(arr)     { const v = Stats.clean(arr); return v.length ? Math.min(...v) : NaN; },
    max(arr)     { const v = Stats.clean(arr); return v.length ? Math.max(...v) : NaN; },

    // ── Higher moments ─────────────────────────────────────────────────────────

    /** Adjusted Fisher-Pearson skewness (matches scipy.stats.skew bias=False) */
    skewness(arr) {
        const v = Stats.clean(arr), n = v.length;
        if (n < 3) return NaN;
        const m = Stats.mean(v), s = Stats.std(v, m);
        if (s === 0) return 0;
        const sum = v.reduce((acc, x) => acc + ((x - m) / s) ** 3, 0);
        return (n / ((n - 1) * (n - 2))) * sum;
    },

    /** Excess kurtosis (Fisher, matches scipy bias=False) */
    kurtosis(arr) {
        const v = Stats.clean(arr), n = v.length;
        if (n < 4) return NaN;
        const m = Stats.mean(v), s = Stats.std(v, m);
        if (s === 0) return 0;
        const sum = v.reduce((acc, x) => acc + ((x - m) / s) ** 4, 0);
        return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum
             - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3));
    },

    // ── Correlation ────────────────────────────────────────────────────────────

    /** Pearson r between two arrays (pairwise NaN exclusion) */
    pearson(xArr, yArr) {
        const n = Math.min(xArr.length, yArr.length);
        const x = [], y = [];
        for (let i = 0; i < n; i++) {
            const xi = typeof xArr[i] === 'string' ? parseFloat(xArr[i]) : Number(xArr[i]);
            const yi = typeof yArr[i] === 'string' ? parseFloat(yArr[i]) : Number(yArr[i]);
            if (isFinite(xi) && isFinite(yi)) { x.push(xi); y.push(yi); }
        }
        if (x.length < 2) return NaN;
        const mx = Stats.mean(x), my = Stats.mean(y);
        let num = 0, dx2 = 0, dy2 = 0;
        for (let i = 0; i < x.length; i++) {
            const dx = x[i] - mx, dy = y[i] - my;
            num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
        }
        if (dx2 === 0 || dy2 === 0) return 0;
        return num / Math.sqrt(dx2 * dy2);
    },

    /** Full correlation matrix */
    correlationMatrix(data, columns) {
        const vectors = columns.map(col => data.map(row => row[col]));
        return columns.map((_, i) =>
            columns.map((_, j) => i === j ? 1 : Stats.pearson(vectors[i], vectors[j]))
        );
    },

    /** Full summary statistics for every column */
    summary(data, columns) {
        return columns.map(col => {
            const vals = Stats.getValues(data, col);
            const m = Stats.mean(vals), s = Stats.std(vals, m);
            return {
                column: col, count: vals.length,
                mean: m, std: s,
                min: Stats.min(vals),
                q25: Stats.quantile(vals, 0.25),
                median: Stats.median(vals),
                q75: Stats.quantile(vals, 0.75),
                max: Stats.max(vals),
                skewness: Stats.skewness(vals),
                kurtosis: Stats.kurtosis(vals),
            };
        });
    },
};
