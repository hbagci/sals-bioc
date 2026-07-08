class Panel {

    /** @param {'left'|'right'} side */
    constructor(side) {
        this.side    = side;
        this.data    = null;   // row-objects returned by the generator
        this.columns = null;   // numeric column names
        this.stats   = null;   // misc run stats from the backend
    }

    // ── DOM refs (always fresh) ────────────────────────────────────────────
    get contentEl()  { return document.getElementById(`${this.side}-content`); }
    get loadingEl()  { return document.getElementById(`${this.side}-loading`); }
    get infoEl()     { return document.getElementById(`${this.side}-stats-info`); }

    // ── Data input ─────────────────────────────────────────────────────────

    setGenerated(result) {
        this.data    = result.rows;
        this.columns = result.columns.filter(c =>
            result.rows.some(r => isFinite(Number(r[c])))
        );
        this.stats = result.stats || {};
    }

    // ── Render ─────────────────────────────────────────────────────────────

    /** realData null = standalone mode (no real CSV loaded yet) */
    async render(realData, realColumns) {
        const show    = Config.show;
        const content = this.contentEl;

        // Purge old Plotly charts
        content.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.purge(el));
        content.innerHTML = '';

        if (!this.data || !this.columns) {
            this._placeholder('Load a CSV file to see visualizations'); return;
        }

        // Badge
        if (this.infoEl) {
            let msg = `${this.data.length} rows · ${this.columns.length} cols`;
            if (this.stats && this.stats.num_unique_states !== undefined) {
                msg += ` · ${this.stats.num_unique_states} states`;
            }
            this.infoEl.textContent = msg;
        }

        const commonCols = (realData && realColumns)
            ? this.columns.filter(c => realColumns.includes(c))
            : null;

        const isComparison = realData && commonCols && commonCols.length > 0;

        if (isComparison) {
            // -- Comparison mode --

            if (commonCols.length < this.columns.length) {
                const note = document.createElement('div');
                note.className = 'note-banner';
                note.textContent = `Comparing ${commonCols.length} of ${this.columns.length} columns (only columns present in both files are compared).`;
                content.appendChild(note);
            }

            const computing = document.createElement('div');
            computing.className = 'computing-msg';
            computing.textContent = 'Computing quality metrics (PCA + t-SNE)…';
            content.appendChild(computing);
            await new Promise(r => setTimeout(r, 30));

            const report = await Metrics.computeReport(realData, this.data, commonCols);
            content.removeChild(computing);

            if (show.scorecard)
                Charts.renderScorecard(content, report, this.side);
            if (show.statsComparison)
                Charts.renderStatsComparison(content, realData, this.data, commonCols);
            if (show.overlaidHistograms)
                Charts.renderOverlaidHistograms(content, realData, this.data, commonCols, this.side);
            if (show.boxPlots)
                Charts.renderComparisonBoxPlots(content, realData, this.data, commonCols, this.side);
            if (show.correlationHeatmaps)
                Charts.renderCorrelationComparison(content, realData, this.data, commonCols, this.side);
            if (show.pcaScatter && commonCols.length >= 2 && report.pcaProjection.real.length > 0)
                Charts.renderPCA(content, report.pcaProjection, this.side, report.pcaScore);
            if (show.tsneScatter && commonCols.length >= 2 && report.tsneProjection.real.length > 0)
                Charts.renderTSNE(content, report.tsneProjection, this.side, report.tsneScore);
            if (show.metricsDetail)
                Charts.renderMetricsTable(content, report);

        } else {
            if (realData && commonCols && commonCols.length === 0) {
                const warn = document.createElement('div');
                warn.className = 'note-banner warn';
                warn.textContent = 'No matching column names between synthetic and real data. Showing standalone analysis.';
                content.appendChild(warn);
            } else if (!realData) {
                const tip = document.createElement('div');
                tip.className = 'note-banner';
                tip.textContent = 'Load real data above for full quality comparison.';
                content.appendChild(tip);
            }

            Charts.renderStandaloneStats(content, this.data, this.columns);
            if (show.overlaidHistograms)
                Charts.renderStandaloneHistograms(content, this.data, this.columns, this.side);
            if (show.boxPlots)
                Charts.renderStandaloneBoxPlots(content, this.data, this.columns, this.side);
            if (show.correlationHeatmaps)
                Charts.renderStandaloneCorrelation(content, this.data, this.columns);
        }
    }

    // ── Clear ──────────────────────────────────────────────────────────────

    clear() {
        const content = this.contentEl;
        content.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.purge(el));
        this.data    = null;
        this.columns = null;
        this.stats   = null;
        this._placeholder('Load a real CSV above and click Generate to see results.');
        if (this.infoEl) this.infoEl.textContent = '';
    }

    _placeholder(msg) {
        this.contentEl.innerHTML = `<div class="placeholder">${msg}</div>`;
    }
}
