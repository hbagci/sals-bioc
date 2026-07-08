class LSTMSide {

    constructor(side) {
        this.side         = side;
        this.csvText      = null;
        this.targetColumn = null;
        this.modelId      = null;

        this.trainParams = {
            epochs: 50, batch_size: 32, learning_rate: 0.003,
            early_stopping_patience: 10, contamination: 0.05,
            train_ratio: 0.8,
            apply_outlier_removal_test: false,
        };

        this.totalRows      = null;
        this.trainResult    = null;
        this.validateResult = null;

        this.trainCsvOverride    = null;
        this.validateEnabled     = false;
        this.validateCsvText     = null;
        this.validateModelB64    = null;

        this.manualCsvText    = null;
        this.manualCsvColumns = null;
        this.manualTargetCol  = null;

        this._trainGenId    = 0;
        this._validateGenId = 0;

        this.el = document.querySelector(`#lstm-section .lstm-side[data-side="${side}"]`);
        this._build();
    }

    // ── UI construction ──────────────────────────────────────────────────────

    _build() {
        this.el.innerHTML = '';

        const color = this.side === 'left' ? 'var(--left)' : 'var(--right)';
        const title = document.createElement('div');
        title.className = 'lstm-side-title';
        title.style.color = color;
        title.textContent = this.side === 'left'
            ? document.getElementById('left-title').textContent
            : document.getElementById('right-title').textContent;
        this.el.appendChild(title);

        this._trainBlock    = this._buildBlock('train',    'Part 1 — Training & Testing',     true);
        this._validateBlock = this._buildBlock('validate', 'Part 2 — Validate',                false);
        this._vizBlock      = this._buildBlock('viz',      'Part 3 — Visualisation & Metrics', true);
        this._manualBlock   = this._buildBlock('manual',   'Part 4 — Manual Predict',          false, false, true);

        this.el.appendChild(this._trainBlock.details);
        this.el.appendChild(this._validateBlock.details);
        this.el.appendChild(this._vizBlock.details);
        this.el.appendChild(this._manualBlock.details);

        this._populateTrainBlock();
        this._populateValidateBlock();
        this._populateVizBlock();
        this._populateManualBlock();
    }

    _buildBlock(part, title, autoDefault, startOpen = true, noControls = false) {
        const details = document.createElement('details');
        details.className = 'lstm-block';
        details.dataset.part = part;
        details.open = startOpen;

        const summary = document.createElement('summary');
        summary.className = 'lstm-block-summary';

        const titleSpan = document.createElement('span');
        titleSpan.className = 'lstm-block-title';
        titleSpan.textContent = title;
        summary.appendChild(titleSpan);

        let autoChk = null, runBtn = null, loadingSpan = null;

        if (!noControls) {
            const controls = document.createElement('span');
            controls.className = 'lstm-block-controls';

            const autoLabel = document.createElement('label');
            autoLabel.className = 'lstm-auto-label';
            autoChk = document.createElement('input');
            autoChk.type = 'checkbox';
            autoChk.className = 'lstm-auto';
            autoChk.dataset.part = part;
            autoChk.checked = autoDefault;
            autoLabel.appendChild(autoChk);
            autoLabel.append(' Auto-run');

            runBtn = document.createElement('button');
            runBtn.className = 'lstm-run-btn';
            runBtn.dataset.part = part;
            runBtn.textContent = 'Run';
            runBtn.addEventListener('click', (e) => { e.preventDefault(); this._onRunClick(part); });

            loadingSpan = document.createElement('span');
            loadingSpan.className = 'lstm-loading';
            loadingSpan.textContent = 'Running…';
            loadingSpan.style.display = 'none';

            controls.appendChild(autoLabel);
            controls.appendChild(runBtn);
            controls.appendChild(loadingSpan);
            summary.appendChild(controls);
        }

        details.appendChild(summary);

        const body = document.createElement('div');
        body.className = 'lstm-block-body';
        details.appendChild(body);

        return { details, body, autoChk, runBtn, loading: loadingSpan };
    }

    // ── Part 1: Training & Testing ───────────────────────────────────────────

    _populateTrainBlock() {
        const body = this._trainBlock.body;

        const csvRow = document.createElement('div');
        csvRow.className = 'lstm-param-row lstm-csv-row';
        csvRow.innerHTML = `<span class="lstm-param-label">CSV source</span>`;

        const csvInfo = document.createElement('span');
        csvInfo.className = 'lstm-csv-info';
        csvInfo.id = `${this.side}-train-csv-info`;
        csvInfo.textContent = 'Using generated CSV';
        csvRow.appendChild(csvInfo);

        const csvFileInput = document.createElement('input');
        csvFileInput.type = 'file'; csvFileInput.accept = '.csv';
        csvFileInput.style.display = 'none';
        csvFileInput.id = `${this.side}-train-csv-file`;
        const csvFileLabel = document.createElement('label');
        csvFileLabel.htmlFor = csvFileInput.id;
        csvFileLabel.className = 'btn-download lstm-file-btn';
        csvFileLabel.textContent = 'Select CSV…';
        csvFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            this.trainCsvOverride = await file.text();
            csvInfo.textContent = file.name;
            const cols = await this._parseColumns(this.trainCsvOverride);
            if (cols) await LSTM.requestTargetColumn(cols, col => { this.targetColumn = col; });
        });
        csvRow.appendChild(csvFileLabel);
        csvRow.appendChild(csvFileInput);

        const clearCsvBtn = document.createElement('button');
        clearCsvBtn.className = 'btn-clear';
        clearCsvBtn.textContent = '✕';
        clearCsvBtn.title = 'Use generated CSV';
        clearCsvBtn.addEventListener('click', () => {
            this.trainCsvOverride = null;
            csvInfo.textContent = 'Using generated CSV';
            csvFileInput.value = '';
        });
        csvRow.appendChild(clearCsvBtn);
        body.appendChild(csvRow);

        const grid = document.createElement('div');
        grid.className = 'lstm-params-grid';
        body.appendChild(grid);

        const paramDefs = [
            { key: 'epochs',                   label: 'Epochs',            type: 'int',   min: 1,    max: 500,  step: 1     },
            { key: 'batch_size',               label: 'Batch size',        type: 'int',   min: 8,    max: 512,  step: 8     },
            { key: 'learning_rate',            label: 'Learning rate',     type: 'float', min: 1e-5, max: 0.1,  step: 0.0001 },
            { key: 'early_stopping_patience',  label: 'ES patience',       type: 'int',   min: 1,    max: 100,  step: 1     },
            { key: 'contamination',            label: 'Contamination',     type: 'float', min: 0.01, max: 0.45, step: 0.01  },
        ];

        for (const p of paramDefs) {
            const f = document.createElement('label');
            f.className = 'lstm-param-field';
            const lbl = document.createElement('span');
            lbl.className = 'lstm-param-label';
            lbl.textContent = p.label;
            f.appendChild(lbl);
            const inp = document.createElement('input');
            inp.type = 'number';
            inp.min = p.min; inp.max = p.max; inp.step = p.step;
            inp.value = this.trainParams[p.key];
            inp.className = 'param-input num';
            inp.addEventListener('input', () => {
                this.trainParams[p.key] = p.type === 'int' ? parseInt(inp.value, 10) : parseFloat(inp.value);
            });
            f.appendChild(inp);
            grid.appendChild(f);
        }

        const isoRow = document.createElement('label');
        isoRow.className = 'lstm-param-field';
        const isoLbl = document.createElement('span');
        isoLbl.className = 'lstm-param-label';
        isoLbl.textContent = 'Outlier removal on test/validate';
        isoLbl.title = 'Apply IsolationForest with the contamination value above to test/validate slices. Off by default.';
        isoRow.appendChild(isoLbl);
        const isoChk = document.createElement('input');
        isoChk.type = 'checkbox';
        isoChk.checked = this.trainParams.apply_outlier_removal_test;
        isoChk.addEventListener('change', () => {
            this.trainParams.apply_outlier_removal_test = isoChk.checked;
        });
        isoRow.appendChild(isoChk);
        grid.appendChild(isoRow);

        const ratioWrap = document.createElement('div');
        ratioWrap.className = 'lstm-ratio-wrap';
        const ratioLabel = document.createElement('span');
        ratioLabel.className = 'lstm-param-label';
        ratioLabel.textContent = 'Train ratio';
        const ratioSlider = document.createElement('input');
        ratioSlider.type = 'range'; ratioSlider.min = 0.1; ratioSlider.max = 0.95; ratioSlider.step = 0.05;
        ratioSlider.value = this.trainParams.train_ratio;
        ratioSlider.className = 'lstm-ratio-slider';
        const ratioDisplay = document.createElement('span');
        ratioDisplay.className = 'lstm-ratio-display';
        ratioDisplay.id = `${this.side}-train-ratio-display`;
        ratioDisplay.textContent = `${this.trainParams.train_ratio}`;
        ratioSlider.addEventListener('input', () => {
            this.trainParams.train_ratio = parseFloat(ratioSlider.value);
            this._updateRatioLabel();
        });
        ratioWrap.appendChild(ratioLabel);
        ratioWrap.appendChild(ratioSlider);
        ratioWrap.appendChild(ratioDisplay);
        body.appendChild(ratioWrap);

        const dlModelBtn = document.createElement('button');
        dlModelBtn.className = 'btn-download';
        dlModelBtn.textContent = 'Download Model (.pt)';
        dlModelBtn.style.display = 'none';
        dlModelBtn.id = `${this.side}-dl-model-btn`;
        body.appendChild(dlModelBtn);

        const resultArea = document.createElement('div');
        resultArea.className = 'lstm-result-area';
        resultArea.id = `${this.side}-train-result`;
        body.appendChild(resultArea);
    }

    _updateRatioLabel() {
        const el = document.getElementById(`${this.side}-train-ratio-display`);
        if (!el) return;
        if (this.totalRows) {
            const trainN = Math.round(this.trainParams.train_ratio * this.totalRows);
            el.textContent = `${this.trainParams.train_ratio} (${trainN} / ${this.totalRows})`;
        } else {
            el.textContent = `${this.trainParams.train_ratio}`;
        }
    }

    // ── Part 2: Validate ─────────────────────────────────────────────────────

    _populateValidateBlock() {
        const body = this._validateBlock.body;

        const enableRow = document.createElement('div');
        enableRow.className = 'lstm-param-row';
        const enableLbl = document.createElement('label');
        enableLbl.className = 'lstm-param-label';
        enableLbl.style.cursor = 'pointer';
        const enableChk = document.createElement('input');
        enableChk.type = 'checkbox';
        enableChk.style.marginRight = '6px';
        enableLbl.appendChild(enableChk);
        enableLbl.append(' Enable validate');
        enableRow.appendChild(enableLbl);
        body.appendChild(enableRow);

        const validateInner = document.createElement('div');
        validateInner.className = 'lstm-validate-inner';
        validateInner.style.display = 'none';
        body.appendChild(validateInner);

        enableChk.addEventListener('change', () => {
            this.validateEnabled = enableChk.checked;
            validateInner.style.display = enableChk.checked ? '' : 'none';
            if (!enableChk.checked) {
                this.validateResult = null;
                this._maybeRunViz();
            }
        });

        const csvRow = document.createElement('div');
        csvRow.className = 'lstm-param-row lstm-csv-row';
        csvRow.innerHTML = `<span class="lstm-param-label">Target CSV (Validate)</span>`;
        const csvInfo = document.createElement('span');
        csvInfo.className = 'lstm-csv-info';
        csvInfo.textContent = 'No file';
        csvRow.appendChild(csvInfo);
        const csvFileInput = document.createElement('input');
        csvFileInput.type = 'file'; csvFileInput.accept = '.csv';
        csvFileInput.style.display = 'none';
        csvFileInput.id = `${this.side}-validate-csv-file`;
        const csvFileLabel = document.createElement('label');
        csvFileLabel.htmlFor = csvFileInput.id;
        csvFileLabel.className = 'btn-download lstm-file-btn';
        csvFileLabel.textContent = 'Select CSV…';
        csvFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            this.validateCsvText = await file.text();
            csvInfo.textContent = file.name;
        });
        csvRow.appendChild(csvFileLabel);
        csvRow.appendChild(csvFileInput);
        const clearCsvBtn = document.createElement('button');
        clearCsvBtn.className = 'btn-clear';
        clearCsvBtn.textContent = '✕';
        clearCsvBtn.addEventListener('click', () => {
            this.validateCsvText = null;
            csvInfo.textContent = 'No file';
            csvFileInput.value = '';
        });
        csvRow.appendChild(clearCsvBtn);
        validateInner.appendChild(csvRow);

        const modelRow = document.createElement('div');
        modelRow.className = 'lstm-param-row lstm-csv-row';
        modelRow.innerHTML = `<span class="lstm-param-label">Model source</span>`;
        const modelInfo = document.createElement('span');
        modelInfo.className = 'lstm-csv-info';
        modelInfo.textContent = 'Using trained model';
        modelRow.appendChild(modelInfo);
        const modelFileInput = document.createElement('input');
        modelFileInput.type = 'file'; modelFileInput.accept = '.pt';
        modelFileInput.style.display = 'none';
        modelFileInput.id = `${this.side}-validate-model-file`;
        const modelFileLabel = document.createElement('label');
        modelFileLabel.htmlFor = modelFileInput.id;
        modelFileLabel.className = 'btn-download lstm-file-btn';
        modelFileLabel.textContent = 'Load .pt…';
        modelFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const buf = await file.arrayBuffer();
            this.validateModelB64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
            modelInfo.textContent = file.name;
        });
        modelRow.appendChild(modelFileLabel);
        modelRow.appendChild(modelFileInput);
        const clearModelBtn = document.createElement('button');
        clearModelBtn.className = 'btn-clear';
        clearModelBtn.textContent = '✕';
        clearModelBtn.addEventListener('click', () => {
            this.validateModelB64 = null;
            modelInfo.textContent = 'Using trained model';
            modelFileInput.value = '';
        });
        modelRow.appendChild(clearModelBtn);
        validateInner.appendChild(modelRow);

        const resultArea = document.createElement('div');
        resultArea.className = 'lstm-result-area';
        resultArea.id = `${this.side}-validate-result`;
        validateInner.appendChild(resultArea);
    }

    // ── Part 3: Viz ──────────────────────────────────────────────────────────

    _populateVizBlock() {
        const body = this._vizBlock.body;
        body.innerHTML = '<div class="lstm-placeholder">Run training first.</div>';
    }

    // ── Part 4: Manual Predict ───────────────────────────────────────────────

    _populateManualBlock() {
        const body = this._manualBlock.body;

        const csvRow = document.createElement('div');
        csvRow.className = 'lstm-param-row lstm-csv-row';
        csvRow.innerHTML = `<span class="lstm-param-label">CSV source</span>`;

        const csvInfo = document.createElement('span');
        csvInfo.className = 'lstm-csv-info';
        csvInfo.id = `${this.side}-manual-csv-info`;
        csvInfo.textContent = 'Using generated CSV';
        csvRow.appendChild(csvInfo);

        const csvFileInput = document.createElement('input');
        csvFileInput.type = 'file'; csvFileInput.accept = '.csv';
        csvFileInput.style.display = 'none';
        csvFileInput.id = `${this.side}-manual-csv-file`;
        const csvFileLabel = document.createElement('label');
        csvFileLabel.htmlFor = csvFileInput.id;
        csvFileLabel.className = 'btn-download lstm-file-btn';
        csvFileLabel.textContent = 'Select CSV…';
        csvFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const text = await file.text();
            const cols = await this._parseColumns(text);
            if (!cols) return;
            await LSTM.requestTargetColumn(cols, col => {
                this.manualTargetCol  = col;
                this.manualCsvColumns = cols;
                this.manualCsvText    = text;
                csvInfo.textContent   = file.name;
                this._buildManualInputTable(cols, text);
            });
        });
        csvRow.appendChild(csvFileLabel);
        csvRow.appendChild(csvFileInput);

        const clearCsvBtn = document.createElement('button');
        clearCsvBtn.className = 'btn-clear';
        clearCsvBtn.textContent = '✕';
        clearCsvBtn.addEventListener('click', () => {
            this.manualCsvText = null; this.manualCsvColumns = null; this.manualTargetCol = null;
            csvInfo.textContent = 'Using generated CSV';
            csvFileInput.value = '';
            const tableArea = body.querySelector('.lstm-manual-table-area');
            if (tableArea) tableArea.innerHTML = '<div class="lstm-placeholder">Select a CSV or fill rows below, then click Predict.</div>';
        });
        csvRow.appendChild(clearCsvBtn);

        const predictBtn = document.createElement('button');
        predictBtn.className = 'btn-generate';
        predictBtn.style.cssText = 'margin-left:auto;padding:4px 14px;font-size:.78rem';
        predictBtn.textContent = 'Predict';
        predictBtn.addEventListener('click', () => this.runPredict());
        csvRow.appendChild(predictBtn);

        body.appendChild(csvRow);

        const tableArea = document.createElement('div');
        tableArea.className = 'lstm-manual-table-area';
        tableArea.innerHTML = '<div class="lstm-placeholder">Select a CSV or fill rows below, then click Predict.</div>';
        body.appendChild(tableArea);

        const resultArea = document.createElement('div');
        resultArea.className = 'lstm-result-area';
        resultArea.id = `${this.side}-manual-result`;
        body.appendChild(resultArea);
    }

    _buildManualInputTable(columns, csvText) {
        const tableArea = this._manualBlock.body.querySelector('.lstm-manual-table-area');
        tableArea.innerHTML = '';

        const featureCols = columns.filter(c => c !== (this.manualTargetCol || this.targetColumn));
        const rows = [];

        const parsedRows = this._quickParseCsv(csvText, columns);
        for (let i = 0; i < 3; i++) {
            rows.push(parsedRows[i] ? [...parsedRows[i]] : featureCols.map(() => 0));
        }

        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        const table = document.createElement('table');
        table.className = 'stats-table lstm-manual-table';
        const thead = table.createTHead();
        const hr = thead.insertRow();
        featureCols.forEach(c => {
            const th = document.createElement('th'); th.textContent = c; hr.appendChild(th);
        });

        this._manualRows = [];
        const tbody = table.createTBody();
        rows.forEach((rowVals, ri) => {
            const tr = tbody.insertRow();
            const rowInputs = [];
            featureCols.forEach((_, ci) => {
                const td = tr.insertCell();
                const inp = document.createElement('input');
                inp.type = 'number'; inp.step = 'any';
                inp.value = isFinite(rowVals[ci]) ? rowVals[ci] : '';
                inp.className = 'param-input num lstm-manual-inp';
                td.appendChild(inp);
                rowInputs.push(inp);
            });
            this._manualRows.push({ inputs: rowInputs, cols: featureCols });
        });

        wrap.appendChild(table);
        tableArea.appendChild(wrap);
    }

    _quickParseCsv(text, columns) {
        const lines = text.trim().split('\n').slice(1, 4);
        return lines.map(line => {
            const vals = line.split(',').map(v => parseFloat(v.trim()));
            return vals;
        });
    }

    // ── Actions ──────────────────────────────────────────────────────────────

    setTargetColumn(col) {
        this.targetColumn = col;
    }

    setSourceCsv(csvText) {
        this.csvText = csvText;
        const infoEl = document.getElementById(`${this.side}-train-csv-info`);
        if (infoEl && !this.trainCsvOverride) infoEl.textContent = 'Using generated CSV';
        if (this._trainBlock.autoChk.checked && this.targetColumn) {
            this.runTrain();
        }
    }

    _onRunClick(part) {
        if (part === 'train')    this.runTrain();
        if (part === 'validate') this.runValidate();
        if (part === 'viz')      this.runViz();
        if (part === 'manual')   this.runPredict();
    }

    async runTrain() {
        const csv = this.trainCsvOverride || this.csvText;
        if (!csv || !this.targetColumn) return;

        const myId = ++this._trainGenId;
        this._setLoading(this._trainBlock, true);
        const resultArea = document.getElementById(`${this.side}-train-result`);
        resultArea.innerHTML = '';

        try {
            const result = await Generator.lstm(Config.api.endpoints.lstmTrain, {
                csv,
                params: { ...this.trainParams },
                target_column: this.targetColumn,
                side: this.side,
            });
            if (myId !== this._trainGenId) return;

            this.modelId     = result.model_id;
            this.totalRows   = result.total_rows;
            this.trainResult = result;
            this._updateRatioLabel();
            resultArea.innerHTML = `<div class="lstm-status-ok">
                ✓ Trained on ${result.train_rows} rows · tested on ${result.test_rows} rows
                (of ${result.total_rows} total) &nbsp;·&nbsp;
                model ID: <code>${result.model_id}</code>
            </div>`;
            this._wireModelDownload(result.model_b64);

            if (this.validateEnabled && this.validateCsvText) this.runValidate();
            if (this._vizBlock.autoChk.checked) this.runViz();

        } catch (err) {
            if (myId !== this._trainGenId) return;
            resultArea.innerHTML = `<div class="error-msg">${err.message}</div>`;
        } finally {
            if (myId === this._trainGenId) this._setLoading(this._trainBlock, false);
        }
    }

    _wireModelDownload(b64) {
        const btn = document.getElementById(`${this.side}-dl-model-btn`);
        if (!btn || !b64) return;
        btn.style.display = 'inline-block';
        btn.onclick = () => {
            const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
            const blob  = new Blob([bytes], { type: 'application/octet-stream' });
            const url   = URL.createObjectURL(blob);
            const a     = document.createElement('a');
            a.href = url; a.download = `lstm_${this.side}.pt`; a.click();
            URL.revokeObjectURL(url);
        };
    }

    async runValidate() {
        if (!this.validateEnabled) return;
        if (!this.modelId && !this.validateModelB64) return;
        if (!this.targetColumn) return;
        if (!this.validateCsvText) return;

        const myId = ++this._validateGenId;
        this._setLoading(this._validateBlock, true);
        const resultArea = document.getElementById(`${this.side}-validate-result`);
        resultArea.innerHTML = '';

        try {
            const result = await Generator.lstm(Config.api.endpoints.lstmValidate, {
                csv:           this.validateCsvText,
                target_column: this.targetColumn,
                side:          this.side,
                model_id:      this.validateModelB64 ? null : this.modelId,
                model_b64:     this.validateModelB64 || null,
                contamination: this.trainParams.contamination,
                apply_outlier_removal_test: this.trainParams.apply_outlier_removal_test,
            });
            if (myId !== this._validateGenId) return;

            this.validateResult = result;
            let msg = `<div class="lstm-status-ok">✓ Validated on ${result.valid_rows} rows</div>`;
            if (result.col_warnings && result.col_warnings.length > 0) {
                msg += `<div class="lifelong-col-warning">${result.col_warnings.map(w => `⚠ ${w}`).join('<br>')}</div>`;
            }
            resultArea.innerHTML = msg;

            if (this._vizBlock.autoChk.checked) this.runViz();

        } catch (err) {
            if (myId !== this._validateGenId) return;
            resultArea.innerHTML = `<div class="error-msg">${err.message}</div>`;
        } finally {
            if (myId === this._validateGenId) this._setLoading(this._validateBlock, false);
        }
    }

    _maybeRunViz() {
        if (this._vizBlock.autoChk.checked) this.runViz();
    }

    runViz() {
        if (!this.trainResult) {
            this._vizBlock.body.innerHTML = '<div class="lstm-placeholder">Run training first.</div>';
            return;
        }
        const body = this._vizBlock.body;
        body.innerHTML = '';

        const train    = this.trainResult.train;
        const test     = this.trainResult.test;
        const validate = (this.validateEnabled && this.validateResult) ? this.validateResult : null;

        Charts.lstmTimeseries(body, train, test, validate);
        Charts.lstmScatter(body, train, test, validate);

        const trainM = Metrics.regression(train.y_true, train.y_pred);
        const testM  = (test.y_true.length > 0) ? Metrics.regression(test.y_true, test.y_pred) : null;
        const valM   = validate ? Metrics.regression(validate.y_true, validate.y_pred) : null;
        this._renderMetricsTable(body, trainM, testM, valM);
    }

    _renderMetricsTable(container, trainM, testM, valM) {
        const sec = document.createElement('div');
        sec.className = 'viz-section';
        sec.innerHTML = '<h3 class="section-title">Regression Metrics</h3>';

        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        const table = document.createElement('table');
        table.className = 'stats-table';
        table.innerHTML = `<thead><tr>
            <th style="text-align:left">Set</th>
            <th>R²</th><th>RMSE</th><th>MAE</th><th>MSE</th>
        </tr></thead>`;
        const tbody = table.createTBody();

        const rows = [['Train', trainM]];
        if (testM) rows.push(['Test', testM]);
        if (valM)  rows.push(['Validate', valM]);

        rows.forEach(([label, m]) => {
            const tr = tbody.insertRow();
            [label,
             isFinite(m.R2)   ? m.R2.toFixed(4)   : '—',
             isFinite(m.RMSE) ? m.RMSE.toFixed(4) : '—',
             isFinite(m.MAE)  ? m.MAE.toFixed(4)  : '—',
             isFinite(m.MSE)  ? m.MSE.toFixed(4)  : '—',
            ].forEach((v, i) => {
                const td = tr.insertCell();
                td.textContent = v;
                if (i === 0) td.className = 'col-name';
            });
        });

        wrap.appendChild(table);
        sec.appendChild(wrap);
        container.appendChild(sec);
    }

    async runPredict() {
        if (!this.modelId && !this.validateModelB64) {
            document.getElementById(`${this.side}-manual-result`).innerHTML =
                '<div class="error-msg">No trained model. Run Part 1 first.</div>';
            return;
        }
        if (!this.targetColumn && !this.manualTargetCol) {
            document.getElementById(`${this.side}-manual-result`).innerHTML =
                '<div class="error-msg">No target column selected.</div>';
            return;
        }

        const targetCol = this.manualTargetCol || this.targetColumn;
        const resultArea = document.getElementById(`${this.side}-manual-result`);
        resultArea.innerHTML = 'Running…';

        const csvSource = this.manualCsvText || this.csvText;
        const cols = this.manualCsvColumns || await this._parseColumns(csvSource);
        if (!cols) { resultArea.innerHTML = '<div class="error-msg">No CSV available.</div>'; return; }

        const featureCols = cols.filter(c => c !== targetCol);

        if (!this._manualRows || this._manualRows.length === 0) {
            this._buildManualInputTable(cols, csvSource);
        }

        const csvRows = (this._manualRows || []).map(r =>
            r.inputs.map((inp, i) => parseFloat(inp.value) || 0)
        );

        try {
            const result = await Generator.lstm(Config.api.endpoints.lstmPred, {
                csv:          csvSource,
                csv_rows:     csvRows,
                columns:      featureCols,
                target_column: targetCol,
                side:         this.side,
                model_id:     this.validateModelB64 ? null : this.modelId,
                model_b64:    this.validateModelB64 || null,
                contamination: this.trainParams.contamination,
            });

            resultArea.innerHTML = '';
            const preds = result.predictions || [];
            const table = document.createElement('table');
            table.className = 'stats-table';
            table.innerHTML = `<thead><tr><th style="text-align:left">Row</th><th>${targetCol} (predicted)</th></tr></thead>`;
            const tbody = table.createTBody();
            preds.forEach((v, i) => {
                const tr = tbody.insertRow();
                tr.insertCell().textContent = i + 1;
                const td = tr.insertCell(); td.textContent = isFinite(v) ? v.toFixed(4) : '—';
                td.style.fontWeight = '700'; td.style.color = 'var(--left)';
            });
            resultArea.appendChild(table);

        } catch (err) {
            resultArea.innerHTML = `<div class="error-msg">${err.message}</div>`;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    _setLoading(block, on) {
        if (block.loading) block.loading.style.display = on ? 'inline' : 'none';
        if (block.runBtn)  block.runBtn.disabled = on;
    }

    async _parseColumns(csvText) {
        return new Promise(resolve => {
            Papa.parse(csvText, {
                header: true, preview: 1,
                complete: res => resolve(res.meta.fields || []),
                error: () => resolve(null),
            });
        });
    }
}

// ── Module ───────────────────────────────────────────────────────────────────

const LSTM = {
    sides: {},
    _sharedTargetColumn: null,

    init() {
        this.sides.left  = new LSTMSide('left');
        this.sides.right = new LSTMSide('right');
    },

    setTargetColumn(col) {
        this._sharedTargetColumn = col;
        if (this.sides.left)  this.sides.left.setTargetColumn(col);
        if (this.sides.right) this.sides.right.setTargetColumn(col);
    },

    setSourceCsv(side, csvText) {
        if (this.sides[side]) this.sides[side].setSourceCsv(csvText);
    },

    requestTargetColumn(columns, callback) {
        return new Promise(resolve => {
            const modal   = document.getElementById('target-column-modal');
            const list    = document.getElementById('modal-col-list');
            const confirm = document.getElementById('modal-confirm');

            list.innerHTML = '';
            let selected = null;

            columns.forEach(col => {
                const label = document.createElement('label');
                label.className = 'modal-col-item';
                const radio = document.createElement('input');
                radio.type = 'radio'; radio.name = 'target-col'; radio.value = col;
                radio.addEventListener('change', () => {
                    selected = col;
                    confirm.disabled = false;
                });
                label.appendChild(radio);
                label.append(' ' + col);
                list.appendChild(label);
            });

            confirm.disabled = true;

            const handler = () => {
                if (selected) {
                    callback(selected);
                    resolve(selected);
                }
                modal.removeEventListener('close', handler);
            };
            modal.addEventListener('close', handler);
            modal.showModal();
        });
    },
};
