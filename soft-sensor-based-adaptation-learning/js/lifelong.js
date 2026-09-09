class LifelongLSTMSide {

    constructor(side, container) {
        this.side      = side;
        this.container = container;

        this._sessionId    = null;
        this._trainResult  = null;
        this._adaptResult  = null;
        this._sourceCsv    = null;
        this._targetCsv    = null;
        this._generatedCsv = null;
        this._targetColumn = null;

        this._params = {
            mu:                      6e-04,
            lam:                     1e-04,
            beta:                    2e-04,
            k:                       5,
            batch_size:              30,
            learning_rate:           1e-3,
            epochs:                  50,
            early_stopping_patience: 10,
            contamination:           0.05,
            train_ratio:             0.8,
            adaptation_batch_size:   30,
            bayes_init_points:       1,
            bayes_n_iter:            1,
            mu_min:  1e-7,  mu_max:  1e-3,
            lam_min: 1e-7,  lam_max: 1e-3,
            beta_min: 1e-9, beta_max: 1e-3,
        };

        this._loadSavedElla();
        this._build();
    }

    _ellaStorageKey() { return `lifelong_ella_last_${this.side}`; }

    _loadSavedElla() {
        try {
            const raw = localStorage.getItem(this._ellaStorageKey());
            if (!raw) return;
            const saved = JSON.parse(raw);
            if (isFinite(saved.mu))   this._params.mu   = saved.mu;
            if (isFinite(saved.lam))  this._params.lam  = saved.lam;
            if (isFinite(saved.beta)) this._params.beta = saved.beta;
        } catch (_) { /* ignore */ }
    }

    _saveElla() {
        try {
            localStorage.setItem(this._ellaStorageKey(), JSON.stringify({
                mu:   this._params.mu,
                lam:  this._params.lam,
                beta: this._params.beta,
            }));
        } catch (_) { /* ignore */ }
    }

    // ── DOM helpers ────────────────────────────────────────────────────────

    _block(title, startOpen) {
        const det = document.createElement('details');
        det.className = 'lstm-block';
        if (startOpen) det.open = true;
        const sum = document.createElement('summary');
        sum.className = 'lstm-block-summary';
        sum.textContent = title;
        det.appendChild(sum);
        const body = document.createElement('div');
        body.className = 'lstm-block-body';
        det.appendChild(body);
        this.container.appendChild(det);
        return { det, body };
    }

    _paramInputsTo(parent, defs, onChange) {
        for (const { label, key, hint } of defs) {
            const row = document.createElement('div');
            row.className = 'lstm-param-row';
            const lbl = document.createElement('label');
            lbl.className = 'lstm-param-label';
            lbl.textContent = label;
            const inp = document.createElement('input');
            inp.type = 'text';
            inp.className = 'lstm-param-input';
            inp.value = this._params[key];
            if (hint) inp.title = hint;
            inp.addEventListener('change', () => {
                const v = parseFloat(inp.value);
                if (isFinite(v)) { this._params[key] = v; inp.value = v; }
                if (onChange) onChange();
            });
            this['_inp_' + key] = inp;
            row.appendChild(lbl);
            row.appendChild(inp);
            parent.appendChild(row);
        }
    }

    _buildFilePicker(parent, label, id, onLoad) {
        const row = document.createElement('div');
        row.className = 'lstm-file-row';
        const lbl = document.createElement('label');
        lbl.className = 'lstm-param-label';
        lbl.textContent = label;
        const input = document.createElement('input');
        input.type = 'file'; input.accept = '.csv';
        input.id = `${this.side}-lifelong-${id}`;
        input.style.display = 'none';
        const fileLabel = document.createElement('label');
        fileLabel.htmlFor = input.id;
        fileLabel.className = 'btn-file lifelong-action-btn';
        fileLabel.textContent = 'Choose CSV';
        const nameSpan = document.createElement('span');
        nameSpan.className = 'file-name';
        nameSpan.textContent = 'No file';
        input.addEventListener('change', async e => {
            const file = e.target.files[0];
            if (!file) return;
            nameSpan.textContent = file.name;
            onLoad(await file.text());
        });
        row.appendChild(lbl);
        row.appendChild(input);
        row.appendChild(fileLabel);
        row.appendChild(nameSpan);
        parent.appendChild(row);
    }

    // ── Build ──────────────────────────────────────────────────────────────

    _build() {
        // ── Block ①: Hyperparameters ──────────────────────────────────────
        const { body: b1 } = this._block('① Hyperparameters', true);

        const ellaGroup = document.createElement('div');
        ellaGroup.className = 'lstm-param-group';
        ellaGroup.innerHTML = '<div class="lstm-group-label">Hyperparameters</div>';
        b1.appendChild(ellaGroup);
        this._paramInputsTo(ellaGroup, [
            { label: 'μ (mu)',     key: 'mu',   hint: 'Memory regularization' },
            { label: 'λ (lambda)', key: 'lam',  hint: 'Basis regularization' },
            { label: 'β (beta)',   key: 'beta', hint: 'Hessian regularization' },
            { label: 'Shared components k', key: 'k', hint: 'Number of shared basis components' },
        ]);

        const saveRow = document.createElement('div');
        saveRow.className = 'lstm-action-row';
        const saveBtn = document.createElement('button');
        saveBtn.className = 'btn-generate lifelong-action-btn';
        saveBtn.textContent = 'Save Params';
        saveBtn.title = 'Persist current μ/λ/β to browser storage so they reload next time';
        const saveStatus = document.createElement('span');
        saveStatus.className = 'loading-indicator';
        saveStatus.style.display = 'none';
        saveBtn.addEventListener('click', () => {
            this._saveElla();
            saveStatus.textContent = 'Saved';
            saveStatus.style.display = 'inline';
            setTimeout(() => { saveStatus.style.display = 'none'; }, 2000);
        });
        saveRow.appendChild(saveBtn);
        saveRow.appendChild(saveStatus);
        ellaGroup.appendChild(saveRow);

        const bayesGroup = document.createElement('div');
        bayesGroup.className = 'lstm-param-group';
        bayesGroup.innerHTML = '<div class="lstm-group-label">BayesOpt search bounds (for Compute button)</div>';
        b1.appendChild(bayesGroup);
        this._paramInputsTo(bayesGroup, [
            { label: 'μ min',              key: 'mu_min' },
            { label: 'μ max',              key: 'mu_max' },
            { label: 'λ min',              key: 'lam_min' },
            { label: 'λ max',              key: 'lam_max' },
            { label: 'β min',              key: 'beta_min' },
            { label: 'β max',              key: 'beta_max' },
            { label: 'BayesOpt init_points', key: 'bayes_init_points' },
            { label: 'BayesOpt n_iter',    key: 'bayes_n_iter' },
        ]);

        const trainGroup = document.createElement('div');
        trainGroup.className = 'lstm-param-group';
        trainGroup.innerHTML = '<div class="lstm-group-label">Training parameters</div>';
        b1.appendChild(trainGroup);
        this._paramInputsTo(trainGroup, [
            { label: 'Batch size',          key: 'batch_size' },
            { label: 'Learning rate',       key: 'learning_rate' },
            { label: 'Epochs',              key: 'epochs' },
            { label: 'Early stop patience', key: 'early_stopping_patience' },
            { label: 'Contamination',       key: 'contamination',  hint: 'IsolationForest outlier fraction' },
            { label: 'Train ratio',         key: 'train_ratio',    hint: 'Source CSV train/test split' },
        ]);

        const computeRow = document.createElement('div');
        computeRow.className = 'lstm-action-row';
        this._computeBtn = document.createElement('button');
        this._computeBtn.className = 'btn-generate lifelong-action-btn';
        this._computeBtn.textContent = 'Compute Optimal (BayesOpt)';
        this._computeStatus = document.createElement('span');
        this._computeStatus.className = 'loading-indicator';
        this._computeStatus.style.display = 'none';
        this._computeStatus.textContent = 'Optimising…';
        computeRow.appendChild(this._computeBtn);
        computeRow.appendChild(this._computeStatus);
        b1.appendChild(computeRow);
        this._computeBtn.addEventListener('click', () => this._runOptimize());

        // ── Block ②: Train ────────────────────────────────────────────────
        const { body: b2 } = this._block('② Train', true);

        // Source CSV row
        const srcRow = document.createElement('div');
        srcRow.className = 'lstm-file-row';
        const srcLbl = document.createElement('label');
        srcLbl.className = 'lstm-param-label';
        srcLbl.textContent = 'Source CSV (Train/Test)';

        const srcInput = document.createElement('input');
        srcInput.type = 'file'; srcInput.accept = '.csv';
        srcInput.id = `${this.side}-lifelong-source`;
        srcInput.style.display = 'none';
        const srcFileBtn = document.createElement('label');
        srcFileBtn.htmlFor = srcInput.id;
        srcFileBtn.className = 'btn-file lifelong-action-btn';
        srcFileBtn.textContent = 'Choose CSV';

        this._useGenBtn = document.createElement('button');
        this._useGenBtn.className = 'btn-generate lifelong-action-btn lifelong-use-gen-btn';
        this._useGenBtn.textContent = 'Use Generated';
        this._useGenBtn.disabled = true;
        this._useGenBtn.title = 'Use the CSV generated by this side above';

        this._srcNameSpan = document.createElement('span');
        this._srcNameSpan.className = 'file-name';
        this._srcNameSpan.textContent = 'No file';

        srcInput.addEventListener('change', async e => {
            const file = e.target.files[0];
            if (!file) return;
            this._srcNameSpan.textContent = file.name;
            this._sourceCsv = await file.text();
            this._parseColumnsFromCsv(this._sourceCsv);
        });
        this._useGenBtn.addEventListener('click', () => {
            this._sourceCsv = this._generatedCsv;
            this._srcNameSpan.textContent = '(generated CSV)';
            this._parseColumnsFromCsv(this._sourceCsv);
        });

        srcRow.appendChild(srcLbl);
        srcRow.appendChild(srcInput);
        srcRow.appendChild(srcFileBtn);
        srcRow.appendChild(this._useGenBtn);
        srcRow.appendChild(this._srcNameSpan);
        b2.appendChild(srcRow);

        // Target column selector
        const colRow = document.createElement('div');
        colRow.className = 'lstm-param-row';
        const colLbl = document.createElement('label');
        colLbl.className = 'lstm-param-label';
        colLbl.textContent = 'Target column';
        this._colSelect = document.createElement('select');
        this._colSelect.className = 'lstm-param-input';
        this._colSelect.style.cursor = 'pointer';
        this._colSelect.addEventListener('change', () => {
            this._targetColumn = this._colSelect.value || null;
            this._updateTrainBtn();
        });
        colRow.appendChild(colLbl);
        colRow.appendChild(this._colSelect);
        b2.appendChild(colRow);

        // Run Train button
        const runTrainRow = document.createElement('div');
        runTrainRow.className = 'lstm-action-row';
        this._runTrainBtn = document.createElement('button');
        this._runTrainBtn.className = 'btn-generate lifelong-action-btn';
        this._runTrainBtn.textContent = 'Run Train';
        this._runTrainBtn.disabled = true;
        this._runTrainStatus = document.createElement('span');
        this._runTrainStatus.className = 'loading-indicator';
        this._runTrainStatus.style.display = 'none';
        this._runTrainStatus.textContent = 'Training…';
        runTrainRow.appendChild(this._runTrainBtn);
        runTrainRow.appendChild(this._runTrainStatus);
        b2.appendChild(runTrainRow);
        this._runTrainBtn.addEventListener('click', () => this._runTrain());

        // ── Block ③: Adapt ────────────────────────────────────────────────
        const { body: b3 } = this._block('③ Adapt', true);

        this._buildFilePicker(b3, 'Target CSV (Adapt)', 'target', csv => {
            this._targetCsv = csv;
            this._updateAdaptBtn();
        });

        // Adaptation batch size
        const adaptBsRow = document.createElement('div');
        adaptBsRow.className = 'lstm-param-row';
        const adaptBsLbl = document.createElement('label');
        adaptBsLbl.className = 'lstm-param-label';
        adaptBsLbl.textContent = 'Adapt batch size';
        const adaptBsInp = document.createElement('input');
        adaptBsInp.type = 'text';
        adaptBsInp.className = 'lstm-param-input';
        adaptBsInp.value = this._params.adaptation_batch_size;
        adaptBsInp.title = 'Rows per prequential step';
        adaptBsInp.addEventListener('change', () => {
            const v = parseInt(adaptBsInp.value);
            if (v > 0) { this._params.adaptation_batch_size = v; adaptBsInp.value = v; }
        });
        this['_inp_adaptation_batch_size'] = adaptBsInp;
        adaptBsRow.appendChild(adaptBsLbl);
        adaptBsRow.appendChild(adaptBsInp);
        b3.appendChild(adaptBsRow);

        // Run Adapt button
        const runAdaptRow = document.createElement('div');
        runAdaptRow.className = 'lstm-action-row';
        this._runAdaptBtn = document.createElement('button');
        this._runAdaptBtn.className = 'btn-generate lifelong-action-btn';
        this._runAdaptBtn.textContent = 'Run Adapt';
        this._runAdaptBtn.disabled = true;
        this._runAdaptStatus = document.createElement('span');
        this._runAdaptStatus.className = 'loading-indicator';
        this._runAdaptStatus.style.display = 'none';
        this._runAdaptStatus.textContent = 'Adapting…';
        runAdaptRow.appendChild(this._runAdaptBtn);
        runAdaptRow.appendChild(this._runAdaptStatus);
        b3.appendChild(runAdaptRow);
        this._runAdaptBtn.addEventListener('click', () => this._runAdapt());

        // ── Block ④: Visualisation ────────────────────────────────────────
        const { det: vizDet, body: vizBody } = this._block('④ Visualisation & Metrics', false);
        this._vizDet   = vizDet;
        this._vizBlock = vizBody;
    }

    // ── Column picker ──────────────────────────────────────────────────────

    _parseColumnsFromCsv(csv) {
        Papa.parse(csv, {
            header: true, dynamicTyping: true, preview: 1,
            complete: res => {
                const cols = (res.meta.fields || []).filter(c =>
                    res.data.length > 0 && typeof res.data[0][c] === 'number'
                );
                this._colSelect.innerHTML = '<option value="">— pick column —</option>';
                cols.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c; opt.textContent = c;
                    if (c === this._targetColumn) opt.selected = true;
                    this._colSelect.appendChild(opt);
                });
                if (!this._targetColumn && cols.length > 0) {
                    this._colSelect.value = cols[0];
                    this._targetColumn = cols[0];
                } else if (this._targetColumn) {
                    this._colSelect.value = this._targetColumn;
                }
                this._updateTrainBtn();
            },
        });
    }

    // ── State helpers ──────────────────────────────────────────────────────

    _updateTrainBtn() {
        this._runTrainBtn.disabled = !(this._sourceCsv && this._targetColumn);
    }

    _updateAdaptBtn() {
        this._runAdaptBtn.disabled = !(this._sessionId && this._targetCsv);
    }

    setTargetColumn(col) {
        this._targetColumn = col;
        if (this._colSelect) this._colSelect.value = col;
        this._updateTrainBtn();
    }

    setGeneratedCsv(csv) {
        this._generatedCsv = csv;
        this._useGenBtn.disabled = false;
    }

    // ── API calls ──────────────────────────────────────────────────────────

    async _runOptimize() {
        if (!this._sourceCsv)    { alert('Load a Source CSV in block ② first.'); return; }
        if (!this._targetColumn) { alert('Select a target column first.'); return; }
        this._computeBtn.disabled = true;
        this._computeStatus.style.display = 'inline';
        try {
            const result = await Generator.lifelong(
                Config.api.endpoints.lifelongOptimize,
                { csv: this._sourceCsv, target_column: this._targetColumn, params: this._params }
            );
            this._params.mu   = result.mu;
            this._params.lam  = result.lam;
            this._params.beta = result.beta;
            if (this['_inp_mu'])   this['_inp_mu'].value   = result.mu;
            if (this['_inp_lam'])  this['_inp_lam'].value  = result.lam;
            if (this['_inp_beta']) this['_inp_beta'].value = result.beta;
            this._computeStatus.textContent = `Done — score: ${result.score.toFixed(4)}`;
            setTimeout(() => {
                this._computeStatus.textContent = 'Optimising…';
                this._computeStatus.style.display = 'none';
            }, 4000);
        } catch (err) {
            alert(`BayesOpt failed: ${err.message}`);
            this._computeStatus.style.display = 'none';
        } finally {
            this._computeBtn.disabled = false;
        }
    }

    async _runTrain() {
        if (!this._sourceCsv || !this._targetColumn) return;
        this._runTrainBtn.disabled = true;
        this._runTrainStatus.style.display = 'inline';
        this._sessionId   = null;
        this._trainResult = null;
        this._adaptResult = null;
        this._updateAdaptBtn();
        this._renderViz();

        try {
            const result = await Generator.lifelong(
                Config.api.endpoints.lifelongTrain,
                {
                    source_csv:    this._sourceCsv,
                    target_column: this._targetColumn,
                    side:          this.side,
                    params:        this._params,
                }
            );
            this._sessionId   = result.session_id;
            this._trainResult = result;
            this._updateAdaptBtn();
            this._renderViz();
        } catch (err) {
            this._vizBlock.innerHTML = `<div class="error-msg"><strong>Train failed:</strong> ${err.message}</div>`;
            this._vizDet.open = true;
        } finally {
            this._runTrainBtn.disabled = !(this._sourceCsv && this._targetColumn);
            this._runTrainStatus.style.display = 'none';
        }
    }

    async _runAdapt() {
        if (!this._sessionId || !this._targetCsv) return;
        this._runAdaptBtn.disabled = true;
        this._runAdaptStatus.style.display = 'inline';
        this._adaptResult = null;
        this._renderViz();

        try {
            const result = await Generator.lifelong(
                Config.api.endpoints.lifelongAdapt,
                {
                    session_id: this._sessionId,
                    target_csv: this._targetCsv,
                    side:       this.side,
                    params:     this._params,
                }
            );
            this._adaptResult = result;
            this._renderViz();
        } catch (err) {
            const errDiv = document.createElement('div');
            errDiv.className = 'error-msg';
            errDiv.innerHTML = `<strong>Adapt failed:</strong> ${err.message}`;
            this._vizBlock.appendChild(errDiv);
            this._vizDet.open = true;
        } finally {
            this._runAdaptBtn.disabled = !(this._sessionId && this._targetCsv);
            this._runAdaptStatus.style.display = 'none';
        }
    }

    // ── Visualisation ──────────────────────────────────────────────────────

    _renderViz() {
        const c = this._vizBlock;
        c.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.purge(el));
        c.innerHTML = '';

        if (!this._trainResult) return;

        const tr = this._trainResult;
        const ad = this._adaptResult;

        if (ad) {
            if (ad.col_warnings && ad.col_warnings.length > 0) {
                const warn = document.createElement('div');
                warn.className = 'lifelong-col-warning';
                warn.innerHTML = ad.col_warnings.map(w => `⚠ ${w}`).join('<br>');
                c.appendChild(warn);
            }
            Charts.lifelongTimeseries(c, tr.train, tr.test, ad.adapt);
            Charts.lifelongBatchLoss(c, ad.batch_metrics, tr.metrics.test.RMSE);
            Charts.lifelongAnimation(c, ad.batch_metrics);
            Charts.lifelongScatter(c, tr.train, tr.test, ad.adapt);
            Charts.lifelongMetricsTable(c, {
                train:             tr.metrics.train,
                test:              tr.metrics.test,
                adapt:             ad.metrics.adapt,
                first_batch_adapt: ad.metrics.first_batch_adapt,
                last_batch_adapt:  ad.metrics.last_batch_adapt,
            });
        } else {
            const emptyAdapt = { y_true: [], y_pred: [] };
            Charts.lifelongTimeseries(c, tr.train, tr.test, emptyAdapt);
            Charts.lifelongScatter(c, tr.train, tr.test, emptyAdapt);
            Charts.lifelongMetricsTable(c, {
                train: tr.metrics.train,
                test:  tr.metrics.test,
            });
        }

        this._vizDet.open = true;
    }
}


// ── Module ─────────────────────────────────────────────────────────────────

const Lifelong = {
    _sides: {},

    init() {
        if (!Config.show.lifelongLSTM) {
            const sec = document.getElementById('lifelong-lstm-section');
            if (sec) sec.style.display = 'none';
            return;
        }
        for (const side of ['left', 'right']) {
            const mount = document.querySelector(`#lifelong-lstm-section .lifelong-side[data-side="${side}"]`);
            if (mount) this._sides[side] = new LifelongLSTMSide(side, mount);
        }
    },

    setTargetColumn(col) {
        for (const s of Object.values(this._sides)) s.setTargetColumn(col);
    },

    setSourceCsv(side, csv) {
        if (this._sides[side]) this._sides[side].setGeneratedCsv(csv);
    },
};
