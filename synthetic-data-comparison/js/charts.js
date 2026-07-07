const Charts = {

    // ── Colour palette ─────────────────────────────────────────────────────────
    //   Real data is always the same neutral tone; synthetic uses the panel colour.
    REAL_COLOR:  '#34495e',
    SYNTH:       { left: '#2980b9', right: '#c0392b' },

    FS(px) {
        const single = document.body.classList.contains('view-ps') ||
                       document.body.classList.contains('view-gcrw');
        return single ? Math.round(px * 1.35) : px;
    },

    get PLOTLY_LAYOUT() {
        return {
            paper_bgcolor: '#ffffff',
            plot_bgcolor:  '#f8f9fb',
            font:          { family: "'Segoe UI', system-ui, sans-serif", size: this.FS(11), color: '#2c3e50' },
            margin:        { l: 50, r: 20, t: 38, b: 45 },
            showlegend:    false,
        };
    },
    PLOTLY_CFG: { responsive: true, displayModeBar: false },

    // ── Helpers ────────────────────────────────────────────────────────────────

    fmt(v, d = 4) {
        if (v === null || v === undefined || !isFinite(v)) return '\u2014';
        return Number(v).toFixed(d);
    },
    pct(v) { return isFinite(v) ? (v * 100).toFixed(1) : '\u2014'; },

    scoreColor(score) {
        if (!isFinite(score)) return '#95a5a6';
        if (score >= 0.80) return '#27ae60';
        if (score >= 0.60) return '#f39c12';
        return '#c0392b';
    },

    /** Create titled section (collapsible) and append to container */
    section(container, title) {
        const sec = document.createElement('details');
        sec.className = 'viz-section';
        sec.open = true;
        const h = document.createElement('summary');
        h.className = 'section-title';
        h.textContent = title;
        sec.appendChild(h);
        container.appendChild(sec);
        return sec;
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // QUALITY SCORECARD  (comparison mode)
    // ═══════════════════════════════════════════════════════════════════════════

    renderScorecard(container, report, side) {
        if (!Config.show.scorecard) return;   // ← visibility from config

        const sec = Charts.section(container, 'Quality Assessment');

        const card = document.createElement('div');
        card.className = 'scorecard';

        // Overall ring
        const ringWrap = document.createElement('div');
        ringWrap.className = 'score-ring-wrap';
        const scorePct  = Math.round(report.overall * 100);
        const ringColor = Charts.scoreColor(report.overall);
        ringWrap.innerHTML = `
            <div class="score-ring" style="background:conic-gradient(${ringColor} ${scorePct}%, #e8ecef 0);">
                <div class="score-ring-inner">
                    <span class="score-number" style="color:${ringColor}">${scorePct}</span>
                    <span class="score-label">Overall</span>
                </div>
            </div>`;
        card.appendChild(ringWrap);

        // Breakdown bars — grouped, weights read dynamically from Config
        const bars = document.createElement('div');
        bars.className = 'score-bars';

        const w = Config.weights;
        const enabled = Config.metrics || {};
        const isOn = k => enabled[k] !== false;
        const effW = k => (isOn(k) ? (w[k] || 0) : 0);
        const totalW = Object.keys(w).reduce((s, k) => s + Math.max(0, effW(k)), 0) || 1;
        const wPct = k => effW(k) > 0 ? Math.round(effW(k) / totalW * 100) + '%' : '—';

        const groups = [
            {
                heading: 'Per-column distributions',
                items: [
                    ['Column Shapes (KS)',  report.columnShapes,   wPct('columnShapes'),   'columnShapes'],
                    ['Distribution (JSD)',  report.avgJSD,         wPct('avgJSD'),         'avgJSD'],
                    ['Earth Mover (W₁)',    report.avgWasserstein, wPct('avgWasserstein'), 'avgWasserstein'],
                    ['Range Coverage',      report.avgRangeCov,    wPct('avgRangeCov'),    'avgRangeCov'],
                    ['Boundary Adherence',  report.avgBoundary,    wPct('avgBoundary'),    'avgBoundary'],
                ],
            },
            {
                heading: 'Correlation structure',
                items: [
                    ['Pair Trends (avg)',       report.pairTrends, wPct('pairTrends'), 'pairTrends'],
                    ['Corr. Frobenius (RMS)',   report.corrFrob,   wPct('corrFrob'),   'corrFrob'],
                ],
            },
            {
                heading: 'Multivariate geometry',
                items: [
                    ['PCA Alignment',    report.pcaScore,   wPct('pcaScore'),  'pcaScore'],
                    ['t-SNE Coverage',   report.tsneScore,  wPct('tsneScore'), 'tsneScore'],
                ],
            },
            {
                heading: 'Diagnostic (not in score)',
                items: [
                    ['Privacy (DCR)', report.privacy, '—', 'privacy'],
                ],
            },
        ];

        groups.forEach(({ heading, items }) => {
            const visibleItems = items.filter(([, , , key]) => isOn(key));
            if (visibleItems.length === 0) return;
            bars.innerHTML += `<div class="bar-group-label">${heading}</div>`;
            visibleItems.forEach(([label, val, weight]) => {
                const pct = isFinite(val) ? Math.round(val * 100) : 0;
                const c   = Charts.scoreColor(val);
                bars.innerHTML += `
                    <div class="bar-row">
                        <span class="bar-label">${label} <span class="bar-weight">${weight}</span></span>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:${pct}%;background:${c}"></div>
                        </div>
                        <span class="bar-val" style="color:${c}">${pct}</span>
                    </div>`;
            });
        });

        card.appendChild(bars);
        sec.appendChild(card);
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // STATS COMPARISON TABLE  (comparison mode — Real vs Synthetic side-by-side)
    // ═══════════════════════════════════════════════════════════════════════════

    renderStatsComparison(container, realData, synthData, columns) {
        const sec = Charts.section(container, 'Summary Statistics — Real vs Synthetic');
        const rStats = Stats.summary(realData, columns);
        const sStats = Stats.summary(synthData, columns);

        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        const table = document.createElement('table');
        table.className = 'stats-table';

        const metrics = ['mean', 'std', 'min', 'median', 'max', 'skewness', 'kurtosis'];
        const mLabels = ['Mean', 'Std', 'Min', 'Median', 'Max', 'Skew', 'Kurt'];

        // Header: Column | metric_real | metric_synth ...
        const thead = table.createTHead();
        const hr = thead.insertRow();
        hr.insertCell().textContent = 'Column';
        mLabels.forEach(l => {
            const thR = document.createElement('th'); thR.textContent = l + ' (R)'; thR.className = 'th-real'; hr.appendChild(thR);
            const thS = document.createElement('th'); thS.textContent = l + ' (S)'; thS.className = 'th-synth'; hr.appendChild(thS);
        });

        const tbody = table.createTBody();
        columns.forEach((col, idx) => {
            const tr = tbody.insertRow();
            const tdName = tr.insertCell();
            tdName.textContent = col;
            tdName.className = 'col-name';
            metrics.forEach(m => {
                const rv = rStats[idx][m], sv = sStats[idx][m];
                const tdR = tr.insertCell(); tdR.textContent = Charts.fmt(rv, 3); tdR.className = 'val-real';
                const tdS = tr.insertCell(); tdS.textContent = Charts.fmt(sv, 3); tdS.className = 'val-synth';
            });
        });

        wrap.appendChild(table);
        sec.appendChild(wrap);
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // OVERLAID HISTOGRAMS  (Real outlined + Synthetic filled, per column)
    // ═══════════════════════════════════════════════════════════════════════════

    renderOverlaidHistograms(container, realData, synthData, columns, side) {
        const sec  = Charts.section(container, 'Distribution Overlay — Real vs Synthetic');
        const grid = document.createElement('div');
        grid.className = 'chart-grid';
        sec.appendChild(grid);

        const synthColor = Charts.SYNTH[side];

        columns.forEach(col => {
            const cell = document.createElement('div');
            cell.className = 'chart-cell';
            grid.appendChild(cell);

            const rVals = Stats.getValues(realData, col);
            const sVals = Stats.getValues(synthData, col);

            // Shared bin edges — derived from the combined range so Real bars
            // look identical regardless of how wide the Synthetic data spreads.
            const allMin  = Math.min(Stats.min(rVals), Stats.min(sVals));
            const allMax  = Math.max(Stats.max(rVals), Stats.max(sVals));
            const nBins   = Math.min(30, Math.max(5, Math.ceil(Math.sqrt(rVals.length + sVals.length))));
            const binSize = (allMax - allMin) / nBins || 1;
            const xbins   = { start: allMin, end: allMax + binSize, size: binSize };

            Plotly.newPlot(cell, [
                {
                    type: 'histogram', x: rVals, xbins, histnorm: 'probability',
                    name: 'Real', opacity: 0.55,
                    marker: { color: Charts.REAL_COLOR + '55', line: { color: Charts.REAL_COLOR, width: 2 } },
                },
                {
                    type: 'histogram', x: sVals, xbins, histnorm: 'probability',
                    name: 'Synthetic', opacity: 0.55,
                    marker: { color: synthColor + '77', line: { color: synthColor, width: 1 } },
                },
            ], {
                ...Charts.PLOTLY_LAYOUT, height: 260, barmode: 'overlay', bargap: 0.04,
                title:  { text: col, font: { size: Charts.FS(11) } },
                xaxis:  { tickfont: { size: Charts.FS(9) }, tickangle: -30 },
                yaxis:  { title: 'Prob.', tickfont: { size: Charts.FS(9) } },
                legend: { x: 0.60, y: 0.95, font: { size: Charts.FS(9) } },
                showlegend: true,
            }, Charts.PLOTLY_CFG);
        });
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPARISON BOX PLOTS  (Real vs Synthetic side-by-side per column)
    // ═══════════════════════════════════════════════════════════════════════════

    renderComparisonBoxPlots(container, realData, synthData, columns, side) {
        const sec  = Charts.section(container, 'Box Plots — Real vs Synthetic');
        const grid = document.createElement('div');
        grid.className = 'chart-grid';
        sec.appendChild(grid);

        const synthColor = Charts.SYNTH[side];
        const bpReal  = realData.length  <= 80 ? 'all' : 'outliers';
        const bpSynth = synthData.length <= 80 ? 'all' : 'outliers';

        columns.forEach(col => {
            const cell = document.createElement('div');
            cell.className = 'chart-cell';
            grid.appendChild(cell);

            const rVals = Stats.getValues(realData, col);
            const sVals = Stats.getValues(synthData, col);

            Plotly.newPlot(cell, [
                {
                    type: 'box', y: rVals, name: 'Real', boxpoints: bpReal, jitter: 0.3,
                    marker: { color: Charts.REAL_COLOR, size: 4, opacity: 0.6 },
                    line: { color: Charts.REAL_COLOR }, fillcolor: Charts.REAL_COLOR + '22',
                },
                {
                    type: 'box', y: sVals, name: 'Synth', boxpoints: bpSynth, jitter: 0.3,
                    marker: { color: synthColor, size: 3, opacity: 0.5 },
                    line: { color: synthColor }, fillcolor: synthColor + '22',
                },
            ], {
                ...Charts.PLOTLY_LAYOUT, height: 270,
                title:  { text: col, font: { size: Charts.FS(11) } },
                yaxis:  { tickfont: { size: Charts.FS(9) } },
                legend: { x: 0.55, y: 0.95, font: { size: Charts.FS(9) } },
                showlegend: true,
            }, Charts.PLOTLY_CFG);
        });
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // CORRELATION COMPARISON  (3 heatmaps: Real, Synthetic, |Difference|)
    // ═══════════════════════════════════════════════════════════════════════════

    renderCorrelationComparison(container, realData, synthData, columns, side) {
        const sec = Charts.section(container, 'Correlation Matrices');

        const corrR = Stats.correlationMatrix(realData, columns);
        const corrS = Stats.correlationMatrix(synthData, columns);
        const diff  = corrR.map((r, i) => r.map((v, j) => Math.abs((v || 0) - (corrS[i][j] || 0))));

        // Shorten column labels for readability
        const shortCols = columns.map(c => c.length > 14 ? c.slice(0, 12) + '…' : c);

        const size = Math.max(350, columns.length * 50 + 100);

        const makeHeatmap = (parentEl, z, title, colorscale, zmin, zmax) => {
            const el = document.createElement('div');
            parentEl.appendChild(el);
            const text = z.map(r => r.map(v => isFinite(v) ? v.toFixed(2) : ''));
            Plotly.newPlot(el, [{
                type: 'heatmap', z, x: shortCols, y: shortCols,
                text, texttemplate: '%{text}',
                textfont: { size: columns.length > 10 ? 8 : 10 },
                colorscale, zmin, zmax,
                colorbar: { thickness: 12, len: 0.85, tickfont: { size: Charts.FS(9) } },
                hovertemplate: '%{y} vs %{x}: %{z:.3f}<extra></extra>',
            }], {
                ...Charts.PLOTLY_LAYOUT, height: size,
                margin: { l: 110, r: 70, t: 36, b: 110 },
                title: { text: title, font: { size: Charts.FS(12), color: '#2c3e50' } },
                xaxis: { tickangle: -45, tickfont: { size: Charts.FS(9) }, side: 'bottom' },
                yaxis: { autorange: 'reversed', tickfont: { size: Charts.FS(9) } },
            }, Charts.PLOTLY_CFG);
        };

        // Row 1: Real + Synthetic side by side
        const topRow = document.createElement('div');
        topRow.className = 'heatmap-row';
        sec.appendChild(topRow);
        const leftCell = document.createElement('div');
        leftCell.className = 'heatmap-cell';
        topRow.appendChild(leftCell);
        const rightCell = document.createElement('div');
        rightCell.className = 'heatmap-cell';
        topRow.appendChild(rightCell);
        makeHeatmap(leftCell,  corrR, 'Real',      'RdBu', -1, 1);
        makeHeatmap(rightCell, corrS, 'Synthetic', 'RdBu', -1, 1);

        // Row 2: Difference full width — green=good (0), red=bad (large diff)
        const diffLabel = document.createElement('div');
        diffLabel.className = 'heatmap-diff-wrap';
        sec.appendChild(diffLabel);
        // Custom green-to-red: 0 = green (match), 1 = red (mismatch)
        const diffColorscale = [
            [0,    '#27ae60'],   // 0.00 — perfect match (green)
            [0.15, '#6fce6f'],   // small diff — light green
            [0.30, '#f4d03f'],   // moderate — yellow
            [0.50, '#f39c12'],   // worse — orange
            [0.75, '#e74c3c'],   // bad — red
            [1,    '#7b241c'],   // terrible — dark red
        ];
        makeHeatmap(diffLabel, diff, '|Difference|  — green = good match, red = poor match',
            diffColorscale, 0, 1);
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // PCA SCATTER  — accepts pre-computed projection from Metrics.computeReport
    // ═══════════════════════════════════════════════════════════════════════════

    renderPCA(container, projection, side, pcaScore) {
        const scoreStr = isFinite(pcaScore) ? ` — Alignment Score: ${Math.round(pcaScore * 100)}` : '';
        const sec = Charts.section(container, `PCA Projection — Real vs Synthetic${scoreStr}`);
        const el  = document.createElement('div');
        el.className = 'chart-full';
        sec.appendChild(el);
        Charts._scatterPlot(el, projection.real, projection.synth, side, 'PC 1', 'PC 2');
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // t-SNE SCATTER  — accepts pre-computed projection from Metrics.computeReport
    // ═══════════════════════════════════════════════════════════════════════════

    renderTSNE(container, projection, side, tsneScore) {
        const scoreStr = isFinite(tsneScore) ? ` — Coverage Score: ${Math.round(tsneScore * 100)}` : '';
        const sec = Charts.section(container, `t-SNE Projection — Real vs Synthetic${scoreStr}`);
        const el  = document.createElement('div');
        el.className = 'chart-full';
        sec.appendChild(el);
        Charts._scatterPlot(el, projection.real, projection.synth, side, 't-SNE 1', 't-SNE 2');
    },

    /** Shared scatter renderer for PCA / t-SNE */
    _scatterPlot(el, real, synth, side, xLabel, yLabel) {
        el.classList.add('proj-scatter');
        const synthColor = Charts.SYNTH[side];
        const singleView = document.body.classList.contains('view-ps') ||
                           document.body.classList.contains('view-gcrw');
        Plotly.newPlot(el, [
            {
                type: 'scattergl', mode: 'markers',
                x: synth.map(p => p.x), y: synth.map(p => p.y),
                name: 'Synthetic', marker: { color: synthColor, size: 5, opacity: 0.4 },
            },
            {
                type: 'scattergl', mode: 'markers',
                x: real.map(p => p.x), y: real.map(p => p.y),
                name: 'Real', marker: { color: Charts.REAL_COLOR, size: 10, opacity: 0.9,
                    line: { color: 'white', width: 1.5 } },
            },
        ], {
            ...Charts.PLOTLY_LAYOUT, height: singleView ? 600 : 520,
            xaxis: { title: xLabel, tickfont: { size: Charts.FS(9) } },
            yaxis: { title: yLabel, tickfont: { size: Charts.FS(9) } },
            legend: { x: 0.01, y: 0.99, font: { size: Charts.FS(10) } },
            showlegend: true,
        }, Charts.PLOTLY_CFG);
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // DETAILED PER-COLUMN METRICS TABLE
    // ═══════════════════════════════════════════════════════════════════════════

    renderMetricsTable(container, report) {
        if (!Config.show.metricsDetail) return;   // ← visibility from config

        const sec = Charts.section(container, 'Detailed Metrics');

        // ── 1. Global / Multivariate metrics ─────────────────────────────────
        const globalSec = document.createElement('div');
        globalSec.style.cssText = 'margin-bottom:16px';
        const globalTitle = document.createElement('div');
        globalTitle.className = 'bar-group-label';
        globalTitle.textContent = 'Global Metrics';
        globalSec.appendChild(globalTitle);

        const globalWrap = document.createElement('div');
        globalWrap.className = 'table-wrap';
        const globalTable = document.createElement('table');
        globalTable.className = 'stats-table metrics-detail';

        const gHead = globalTable.createTHead();
        const gHr = gHead.insertRow();
        ['Metric', 'Score', 'Score %', 'Weight in Overall'].forEach(h => {
            const th = document.createElement('th'); th.textContent = h; gHr.appendChild(th);
        });

        const w = Config.weights;
        const totalW = Object.values(w).reduce((s, v) => s + (v > 0 ? v : 0), 0) || 1;

        const globalRows = [
            ['PCA Alignment',       report.pcaScore,        'pcaScore'],
            ['t-SNE Coverage',      report.tsneScore,       'tsneScore'],
            ['Corr. Pair Trends',   report.pairTrends,      'pairTrends'],
            ['Corr. Frobenius RMS', report.corrFrob,        'corrFrob'],
            ['Column Shapes (KS)',  report.columnShapes,    'columnShapes'],
            ['Distribution (JSD)', report.avgJSD,           'avgJSD'],
            ['Earth Mover (W₁)',   report.avgWasserstein,   'avgWasserstein'],
            ['Range Coverage',     report.avgRangeCov,      'avgRangeCov'],
            ['Boundary Adherence', report.avgBoundary,      'avgBoundary'],
            ['Privacy (DCR)',       report.privacy,         null],   // diagnostic only
        ];

        const gBody = globalTable.createTBody();
        globalRows.forEach(([label, val, wKey]) => {
            const tr = gBody.insertRow();
            const tdName = tr.insertCell(); tdName.textContent = label; tdName.className = 'col-name';
            const tdRaw  = tr.insertCell(); tdRaw.textContent  = Charts.fmt(val, 4);
            const tdPct  = tr.insertCell();
            tdPct.textContent  = Charts.pct(val) + '%';
            tdPct.style.color  = Charts.scoreColor(val);
            tdPct.style.fontWeight = '600';
            const tdW = tr.insertCell();
            if (wKey && w[wKey] > 0) {
                tdW.textContent   = Math.round(w[wKey] / totalW * 100) + '%';
                tdW.style.color   = '#7f8c8d';
            } else {
                tdW.textContent   = wKey ? '0% (disabled)' : '— (diagnostic)';
                tdW.style.color   = '#b0b8c1';
                tdW.style.fontStyle = 'italic';
            }
        });

        globalWrap.appendChild(globalTable);
        globalSec.appendChild(globalWrap);
        sec.appendChild(globalSec);

        // ── 2. Per-column metrics ─────────────────────────────────────────────
        const colTitle = document.createElement('div');
        colTitle.className = 'bar-group-label';
        colTitle.textContent = 'Per-Column Metrics';
        sec.appendChild(colTitle);

        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        const table = document.createElement('table');
        table.className = 'stats-table metrics-detail';

        const thead = table.createTHead();
        const hr = thead.insertRow();
        ['Column', 'KS Score', 'JSD Score', 'Wasserstein', 'Range Cov.', 'Boundary'].forEach(h => {
            const th = document.createElement('th'); th.textContent = h; hr.appendChild(th);
        });

        const tbody = table.createTBody();
        report.columnMetrics.forEach(m => {
            const tr = tbody.insertRow();
            const vals = [m.column, m.ks, m.jsd, m.wasserstein, m.rangeCov, m.boundary];
            vals.forEach((v, i) => {
                const td = tr.insertCell();
                if (i === 0) {
                    td.textContent = v;
                    td.className = 'col-name';
                } else {
                    td.textContent = Charts.pct(v) + '%';
                    td.style.color = Charts.scoreColor(v);
                    td.style.fontWeight = '600';
                }
            });
        });

        wrap.appendChild(table);
        sec.appendChild(wrap);
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // STANDALONE MODE  (no real data — basic individual analysis)
    // ═══════════════════════════════════════════════════════════════════════════

    renderStandaloneStats(container, data, columns) {
        const sec = Charts.section(container, 'Summary Statistics');
        const stats = Stats.summary(data, columns);
        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        const table = document.createElement('table');
        table.className = 'stats-table';
        const headers = ['Column', 'N', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max', 'Skew', 'Kurt'];
        const thead = table.createTHead();
        const hr = thead.insertRow();
        headers.forEach(h => { const th = document.createElement('th'); th.textContent = h; hr.appendChild(th); });
        const tbody = table.createTBody();
        stats.forEach(s => {
            const tr = tbody.insertRow();
            [s.column, s.count, Charts.fmt(s.mean,4), Charts.fmt(s.std,4), Charts.fmt(s.min,4),
             Charts.fmt(s.q25,4), Charts.fmt(s.median,4), Charts.fmt(s.q75,4), Charts.fmt(s.max,4),
             Charts.fmt(s.skewness,3), Charts.fmt(s.kurtosis,3)].forEach((v, i) => {
                const td = tr.insertCell(); td.textContent = v;
                if (i === 0) td.className = 'col-name';
            });
        });
        wrap.appendChild(table);
        sec.appendChild(wrap);
    },

    renderStandaloneHistograms(container, data, columns, side) {
        const sec  = Charts.section(container, 'Distributions');
        const grid = document.createElement('div'); grid.className = 'chart-grid'; sec.appendChild(grid);
        const color = Charts.SYNTH[side];
        columns.forEach(col => {
            const cell = document.createElement('div'); cell.className = 'chart-cell'; grid.appendChild(cell);
            const vals = Stats.getValues(data, col);
            const bins = Math.min(30, Math.max(5, Math.ceil(Math.sqrt(vals.length))));
            Plotly.newPlot(cell, [{
                type: 'histogram', x: vals, nbinsx: bins, histnorm: 'probability',
                marker: { color, opacity: 0.80, line: { color: 'white', width: 0.8 } },
            }], {
                ...Charts.PLOTLY_LAYOUT, height: 260, bargap: 0.04,
                title: { text: col, font: { size: Charts.FS(11) } },
                xaxis: { tickfont: { size: Charts.FS(9) }, tickangle: -30 },
                yaxis: { title: 'Prob.', tickfont: { size: Charts.FS(9) } },
            }, Charts.PLOTLY_CFG);
        });
    },

    renderStandaloneBoxPlots(container, data, columns, side) {
        const sec  = Charts.section(container, 'Box Plots');
        const grid = document.createElement('div'); grid.className = 'chart-grid'; sec.appendChild(grid);
        const color = Charts.SYNTH[side];
        const bp = data.length <= 80 ? 'all' : 'outliers';
        columns.forEach(col => {
            const cell = document.createElement('div'); cell.className = 'chart-cell'; grid.appendChild(cell);
            const vals = Stats.getValues(data, col);
            Plotly.newPlot(cell, [{
                type: 'box', y: vals, name: col, boxpoints: bp, jitter: 0.35,
                marker: { color, size: 4, opacity: 0.6 }, line: { color }, fillcolor: color + '33',
            }], {
                ...Charts.PLOTLY_LAYOUT, height: 260,
                title: { text: col, font: { size: Charts.FS(11) } },
                xaxis: { visible: false }, yaxis: { tickfont: { size: Charts.FS(9) } },
            }, Charts.PLOTLY_CFG);
        });
    },

    renderStandaloneCorrelation(container, data, columns) {
        const sec = Charts.section(container, 'Correlation Heatmap');
        const el  = document.createElement('div'); el.className = 'chart-full'; sec.appendChild(el);
        const matrix = Stats.correlationMatrix(data, columns);
        const z    = matrix.map(r => r.map(v => parseFloat((v||0).toFixed(4))));
        const text = z.map(r => r.map(v => v.toFixed(2)));
        const size = Math.max(350, columns.length * 52 + 80);
        Plotly.newPlot(el, [{
            type: 'heatmap', z, x: columns, y: columns, text, texttemplate: '%{text}',
            textfont: { size: Charts.FS(10) }, colorscale: 'RdBu', reversescale: true, zmin: -1, zmax: 1,
            colorbar: { thickness: 14, len: 0.9 },
        }], {
            ...Charts.PLOTLY_LAYOUT, height: size,
            margin: { l: 130, r: 80, t: 30, b: 130 },
            xaxis: { tickangle: -45, tickfont: { size: Charts.FS(10) } },
            yaxis: { autorange: 'reversed', tickfont: { size: Charts.FS(10) } },
        }, Charts.PLOTLY_CFG);
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // BUCKETING ITERATIONS  (EMCM-PS only — right side shows content, left stays blank)
    // ═══════════════════════════════════════════════════════════════════════════

    renderBucketingIterations(container, trace) {
        if (!Config.show.bucketingIterations) return;

        const sec = document.createElement('details');
        sec.className = 'viz-section bucketing-iter-section';
        sec.open = false;

        const summary = document.createElement('summary');
        summary.className = 'section-title bucketing-iter-summary';
        summary.textContent = 'Adaptive Bucketing — Recursive Largest-Gap Splits';
        sec.appendChild(summary);

        const body = document.createElement('div');
        body.className = 'bucketing-iter-body';

        if (!trace || trace.length === 0) {
            body.innerHTML = '<div class="bucketing-iter-blank">— Not applicable for this algorithm —</div>';
            sec.appendChild(body);
            container.appendChild(sec);
            return;
        }

        const final = trace[trace.length - 1];
        const finalRatio = final.final_ratio !== undefined ? final.final_ratio : final.ratio;
        const targetRatio = final.target_ratio;
        const unreachable = !!final.unreachable;
        const nSplits   = final.n_splits    !== undefined ? final.n_splits    : trace.length - 1;
        const nAccepted = final.n_accepted  !== undefined ? final.n_accepted  : trace.filter(s => s.phase !== 'final' && s.accepted).length;

        const banner = document.createElement('div');
        banner.className = 'bucketing-iter-banner';
        if (unreachable) {
            banner.innerHTML = `<strong>Target not reachable</strong> — tried ${nSplits} splits (${nAccepted} accepted), final state ratio ${finalRatio} (target ≤ ${targetRatio}). Each step pops the largest remaining gap (relative to its column's range) and cuts there, creating a <em>ghost bucket</em> in the empty region. A split is accepted only if the new joint-state ratio stays within the target.`;
            banner.classList.add('warn');
        } else {
            banner.innerHTML = `<strong>Converged</strong> — ${nAccepted} / ${nSplits} splits accepted, final state ratio ${finalRatio} (target ≤ ${targetRatio}). Each step pops the largest remaining gap (relative to its column's range) and cuts there, creating a <em>ghost bucket</em> in the empty region. A split is accepted only if the new joint-state ratio stays within the target.`;
        }
        body.appendChild(banner);

        // Final per-column edges
        if (final.final_edges) {
            const edgesWrap = document.createElement('div');
            edgesWrap.className = 'bucketing-iter-edges';
            edgesWrap.innerHTML = '<div class="moves-title">Final bucket edges per column</div>';
            const etbl = document.createElement('table');
            etbl.className = 'stats-table bucketing-iter-edges-table';
            etbl.innerHTML = '<thead><tr><th>Column</th><th># Buckets</th><th>Edges</th></tr></thead>';
            const etb = document.createElement('tbody');
            Object.entries(final.final_edges).forEach(([col, edges]) => {
                const nb = edges.length - 1;
                const edgesStr = edges.map(e => e.toFixed(4)).join(', ');
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${col}</td><td>${nb}</td><td class="edges-cell">${edgesStr}</td>`;
                etb.appendChild(tr);
            });
            etbl.appendChild(etb);
            edgesWrap.appendChild(etbl);
            body.appendChild(edgesWrap);
        }

        sec.appendChild(body);
        container.appendChild(sec);
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // LSTM CHARTS
    // ═══════════════════════════════════════════════════════════════════════════

    lstmTimeseries(container, train, test, validate) {
        const el = document.createElement('div');
        el.className = 'lstm-chart-cell';
        container.appendChild(el);

        const trainLen = train.y_true.length;
        const testLen  = test ? test.y_true.length : 0;
        const valLen   = validate ? validate.y_true.length : 0;

        const yTrue = [...train.y_true, ...(test ? test.y_true : []), ...(validate ? validate.y_true : [])];
        const yPred = [...train.y_pred, ...(test ? test.y_pred : []), ...(validate ? validate.y_pred : [])];

        const trainEnd = trainLen;
        const testEnd  = trainEnd + testLen;
        const valEnd   = testEnd  + valLen;

        const shapes = [
            { type: 'rect', x0: 0, x1: trainEnd, y0: 0, y1: 1, yref: 'paper',
              fillcolor: 'LightSkyBlue', opacity: 0.18, line: { width: 0 } },
        ];
        const annotations = [
            { x: trainEnd / 2, y: 1.06, yref: 'paper', xanchor: 'center',
              text: `Train (${trainLen})`, showarrow: false, font: { size: Charts.FS(10), color: '#2471a3' } },
        ];

        if (testLen > 0) {
            shapes.push(
                { type: 'rect', x0: trainEnd, x1: testEnd, y0: 0, y1: 1, yref: 'paper',
                  fillcolor: 'LightGreen', opacity: 0.18, line: { width: 0 } },
                { type: 'line', x0: trainEnd, x1: trainEnd, y0: 0, y1: 1, yref: 'paper',
                  line: { color: 'black', width: 1.5, dash: 'dash' } },
            );
            annotations.push(
                { x: trainEnd + testLen / 2, y: 1.06, yref: 'paper', xanchor: 'center',
                  text: `Test (${testLen})`, showarrow: false, font: { size: Charts.FS(10), color: '#1e8449' } },
            );
        }
        if (valLen > 0) {
            shapes.push(
                { type: 'rect', x0: testEnd, x1: valEnd, y0: 0, y1: 1, yref: 'paper',
                  fillcolor: 'LightCoral', opacity: 0.18, line: { width: 0 } },
                { type: 'line', x0: testEnd, x1: testEnd, y0: 0, y1: 1, yref: 'paper',
                  line: { color: '#c0392b', width: 1.5, dash: 'dash' } },
            );
            annotations.push(
                { x: testEnd + valLen / 2, y: 1.06, yref: 'paper', xanchor: 'center',
                  text: `Validate (${valLen})`, showarrow: false, font: { size: Charts.FS(10), color: '#922b21' } },
            );
        }

        Plotly.newPlot(el, [
            { y: yTrue, mode: 'lines', name: 'True',      line: { color: '#2980b9' } },
            { y: yPred, mode: 'lines', name: 'Predicted', line: { color: '#e67e22', dash: 'dash' } },
        ], {
            ...Charts.PLOTLY_LAYOUT, height: 350, showlegend: true,
            title:  { text: 'LSTM Performance — True vs Predicted', font: { size: Charts.FS(12) } },
            xaxis:  { tickfont: { size: Charts.FS(9) } },
            yaxis:  { tickfont: { size: Charts.FS(9) } },
            legend: { x: 0.70, y: 0.99, font: { size: Charts.FS(9) } },
            shapes,
            annotations,
        }, Charts.PLOTLY_CFG);
    },

    lstmScatter(container, train, test, validate) {
        const el = document.createElement('div');
        el.className = 'lstm-chart-cell';
        container.appendChild(el);

        const allTrue = [
            ...train.y_true,
            ...(test ? test.y_true : []),
            ...(validate ? validate.y_true : []),
        ];
        const lo = Math.min(...allTrue), hi = Math.max(...allTrue);

        const traces = [
            { type: 'scattergl', mode: 'markers', name: 'Train',
              x: train.y_true, y: train.y_pred,
              marker: { color: '#2980b9', size: 4, opacity: 0.5 } },
        ];
        if (test && test.y_true.length > 0) {
            traces.push({ type: 'scattergl', mode: 'markers', name: 'Test',
                x: test.y_true, y: test.y_pred,
                marker: { color: '#27ae60', size: 4, opacity: 0.5 } });
        }
        if (validate && validate.y_true.length > 0) {
            traces.push({ type: 'scattergl', mode: 'markers', name: 'Validate',
                x: validate.y_true, y: validate.y_pred,
                marker: { color: '#e74c3c', size: 4, opacity: 0.5 } });
        }
        traces.push({ type: 'scatter', mode: 'lines', name: 'Perfect fit',
            x: [lo, hi], y: [lo, hi], showlegend: false,
            line: { color: '#2c3e50', dash: 'dash', width: 1.5 } });

        Plotly.newPlot(el, traces, {
            ...Charts.PLOTLY_LAYOUT, height: 360, showlegend: true,
            title:  { text: 'Predicted vs True', font: { size: Charts.FS(12) } },
            xaxis:  { title: 'True', tickfont: { size: Charts.FS(9) } },
            yaxis:  { title: 'Predicted', tickfont: { size: Charts.FS(9) } },
            legend: { x: 0.01, y: 0.99, font: { size: Charts.FS(9) } },
        }, Charts.PLOTLY_CFG);
    },

    lifelongTimeseries(container, train, test, adapt) {
        const el = document.createElement('div');
        el.className = 'lstm-chart-cell';
        container.appendChild(el);

        const yTrue  = [...train.y_true, ...test.y_true, ...adapt.y_true];
        const yPred  = [...train.y_pred, ...test.y_pred, ...adapt.y_pred];
        const trainEnd = train.y_true.length;
        const testEnd  = trainEnd + test.y_true.length;

        Plotly.newPlot(el, [
            { y: yTrue, mode: 'lines', name: 'True',      line: { color: '#2980b9' } },
            { y: yPred, mode: 'lines', name: 'Predicted', line: { color: '#e67e22', dash: 'dash' } },
        ], {
            ...Charts.PLOTLY_LAYOUT, height: 380, showlegend: true,
            title:  { text: 'Lifelong LSTM — Train / Test / Adapt', font: { size: Charts.FS(12) } },
            xaxis:  { tickfont: { size: Charts.FS(9) } },
            yaxis:  { tickfont: { size: Charts.FS(9) } },
            legend: { x: 0.70, y: 0.99, font: { size: Charts.FS(9) } },
            shapes: [
                { type: 'rect', x0: 0,         x1: trainEnd,      y0: 0, y1: 1, yref: 'paper', fillcolor: 'LightSkyBlue', opacity: 0.18, line: { width: 0 } },
                { type: 'rect', x0: trainEnd,   x1: testEnd,       y0: 0, y1: 1, yref: 'paper', fillcolor: 'LightGreen',  opacity: 0.18, line: { width: 0 } },
                { type: 'rect', x0: testEnd,    x1: yTrue.length,  y0: 0, y1: 1, yref: 'paper', fillcolor: 'LightCoral',  opacity: 0.18, line: { width: 0 } },
                { type: 'line', x0: trainEnd, x1: trainEnd, y0: 0, y1: 1, yref: 'paper', line: { color: 'black', width: 1.5, dash: 'dash' } },
                { type: 'line', x0: testEnd,  x1: testEnd,  y0: 0, y1: 1, yref: 'paper', line: { color: '#c0392b', width: 1.5, dash: 'dash' } },
            ],
            annotations: [
                { x: trainEnd / 2,                       y: 1.07, yref: 'paper', xanchor: 'center', showarrow: false, text: `Train (${trainEnd})`,              font: { size: Charts.FS(10), color: '#2471a3' } },
                { x: trainEnd + test.y_true.length / 2,  y: 1.07, yref: 'paper', xanchor: 'center', showarrow: false, text: `Test (${test.y_true.length})`,     font: { size: Charts.FS(10), color: '#1e8449' } },
                { x: testEnd  + adapt.y_true.length / 2, y: 1.07, yref: 'paper', xanchor: 'center', showarrow: false, text: `Adapt (${adapt.y_true.length})`,   font: { size: Charts.FS(10), color: '#922b21' } },
            ],
        }, Charts.PLOTLY_CFG);
    },

    lifelongBatchLoss(container, batchMetrics, testRmse) {
        const el = document.createElement('div');
        el.className = 'lstm-chart-cell';
        container.appendChild(el);

        const xs   = batchMetrics.map((_, i) => i + 1);
        const rmse = batchMetrics.map(b => b.rmse);

        Plotly.newPlot(el, [
            { x: xs, y: rmse, mode: 'lines+markers', name: 'RMSE', line: { color: '#e74c3c' }, marker: { size: 5 } },
            isFinite(testRmse) ? {
                x: [1, xs.length], y: [testRmse, testRmse], mode: 'lines', name: 'Pre-adapt RMSE',
                line: { color: '#7f8c8d', dash: 'dot', width: 1.5 },
            } : null,
        ].filter(Boolean), {
            ...Charts.PLOTLY_LAYOUT, height: 300, showlegend: true,
            title: { text: 'Adaptation Loss per Batch', font: { size: Charts.FS(12) } },
            xaxis: { title: 'Batch', tickfont: { size: Charts.FS(9) } },
            yaxis: { title: 'RMSE', tickfont: { size: Charts.FS(9) } },
            legend: { x: 0.70, y: 0.99, font: { size: Charts.FS(9) } },
        }, Charts.PLOTLY_CFG);
    },

    lifelongAnimation(container, batchMetrics) {
        if (!batchMetrics || batchMetrics.length === 0) return;

        const allVals = batchMetrics.flatMap(b => [...b.y_true, ...b.y_pred]);
        const yMin = Math.min(...allVals), yMax = Math.max(...allVals);
        const pad  = Math.abs(yMax - yMin) * 0.05 || 1;

        const wrapper = document.createElement('div');
        wrapper.className = 'lstm-chart-cell';
        container.appendChild(wrapper);

        const ctrlRow = document.createElement('div');
        ctrlRow.className = 'lifelong-anim-controls';

        const playBtn = document.createElement('button');
        playBtn.className = 'lifelong-anim-play';
        playBtn.textContent = '▶ Play';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = 0;
        slider.max = batchMetrics.length - 1;
        slider.value = 0;
        slider.className = 'lifelong-anim-slider';

        const lbl = document.createElement('span');
        lbl.className = 'lifelong-anim-label';
        lbl.textContent = `Batch 1 / ${batchMetrics.length}`;

        ctrlRow.appendChild(playBtn);
        ctrlRow.appendChild(slider);
        ctrlRow.appendChild(lbl);
        wrapper.appendChild(ctrlRow);

        const el = document.createElement('div');
        wrapper.appendChild(el);

        const layout = {
            ...Charts.PLOTLY_LAYOUT, height: 300, showlegend: true,
            title:  { text: 'Adaptation Animation — Batch-by-Batch', font: { size: Charts.FS(12) } },
            xaxis:  { tickfont: { size: Charts.FS(9) } },
            yaxis:  { range: [yMin - pad, yMax + pad], tickfont: { size: Charts.FS(9) } },
            legend: { x: 0.70, y: 0.99, font: { size: Charts.FS(9) } },
        };

        const traces = () => {
            const b = batchMetrics[+slider.value];
            return [
                { y: b.y_true, mode: 'lines', name: 'True',      line: { color: '#2980b9' } },
                { y: b.y_pred, mode: 'lines', name: 'Predicted', line: { color: '#e67e22', dash: 'dash' } },
            ];
        };

        Plotly.newPlot(el, traces(), layout, Charts.PLOTLY_CFG);

        slider.addEventListener('input', () => {
            lbl.textContent = `Batch ${+slider.value + 1} / ${batchMetrics.length}`;
            Plotly.react(el, traces(), layout, Charts.PLOTLY_CFG);
        });

        let playing = false;
        let timer   = null;

        playBtn.addEventListener('click', () => {
            if (playing) {
                clearInterval(timer);
                playing = false;
                playBtn.textContent = '▶ Play';
                return;
            }
            if (+slider.value >= batchMetrics.length - 1) slider.value = 0;
            playing = true;
            playBtn.textContent = '⏸ Pause';
            timer = setInterval(() => {
                const next = +slider.value + 1;
                if (next >= batchMetrics.length) {
                    clearInterval(timer);
                    playing = false;
                    playBtn.textContent = '▶ Play';
                    return;
                }
                slider.value = next;
                slider.dispatchEvent(new Event('input'));
            }, 300);
        });
    },

    lifelongMetricsTable(container, metrics) {
        const el = document.createElement('div');
        el.className = 'lstm-chart-cell';
        container.appendChild(el);

        const phases = [
            ['Train',               metrics.train],
            ['Test',                metrics.test],
            ['Adapt (overall)',     metrics.adapt],
            ['Adapt (first batch)', metrics.first_batch_adapt],
            ['Adapt (last batch)',  metrics.last_batch_adapt],
        ];
        const fmt = v => (v === undefined || v === null || !isFinite(v)) ? '—' : Number(v).toFixed(4);

        let html = '<table class="lstm-metrics-table"><thead><tr><th>Phase</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MSE</th></tr></thead><tbody>';
        for (const [label, m] of phases) {
            if (!m) continue;
            html += `<tr><td>${label}</td><td>${fmt(m.R2)}</td><td>${fmt(m.RMSE)}</td><td>${fmt(m.MAE)}</td><td>${fmt(m.MSE)}</td></tr>`;
        }
        html += '</tbody></table>';
        el.innerHTML = html;
    },

    lifelongScatter(container, train, test, adapt) {
        const el = document.createElement('div');
        el.className = 'lstm-chart-cell';
        container.appendChild(el);

        const allTrue = [...train.y_true, ...test.y_true, ...adapt.y_true];
        const lo = Math.min(...allTrue), hi = Math.max(...allTrue);

        Plotly.newPlot(el, [
            { type: 'scattergl', mode: 'markers', name: 'Train',  x: train.y_true,  y: train.y_pred,  marker: { color: '#2980b9', size: 4, opacity: 0.5 } },
            { type: 'scattergl', mode: 'markers', name: 'Test',   x: test.y_true,   y: test.y_pred,   marker: { color: '#27ae60', size: 4, opacity: 0.5 } },
            { type: 'scattergl', mode: 'markers', name: 'Adapt',  x: adapt.y_true,  y: adapt.y_pred,  marker: { color: '#e74c3c', size: 4, opacity: 0.5 } },
            { type: 'scatter',   mode: 'lines',   name: 'Perfect', x: [lo, hi], y: [lo, hi], showlegend: false, line: { color: '#2c3e50', dash: 'dash', width: 1.5 } },
        ], {
            ...Charts.PLOTLY_LAYOUT, height: 380, showlegend: true,
            title: { text: 'Predicted vs True (all phases)', font: { size: Charts.FS(12) } },
            xaxis: { title: 'True', tickfont: { size: Charts.FS(9) } },
            yaxis: { title: 'Predicted', tickfont: { size: Charts.FS(9) } },
            legend: { x: 0.01, y: 0.99, font: { size: Charts.FS(9) } },
        }, Charts.PLOTLY_CFG);
    },

    /** Render an empty placeholder so the opposite side keeps visual alignment. */
    renderBucketingIterationsBlank(container) {
        if (!Config.show.bucketingIterations) return;
        const sec = document.createElement('details');
        sec.className = 'viz-section bucketing-iter-section blank';
        sec.open = false;
        const summary = document.createElement('summary');
        summary.className = 'section-title bucketing-iter-summary';
        summary.textContent = 'Adaptive Bucketing — Iteration Trace';
        sec.appendChild(summary);
        const body = document.createElement('div');
        body.className = 'bucketing-iter-body';
        body.innerHTML = '<div class="bucketing-iter-blank">— Not applicable for this algorithm —</div>';
        sec.appendChild(body);
        container.appendChild(sec);
    },
};
