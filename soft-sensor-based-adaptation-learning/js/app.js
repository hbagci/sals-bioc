document.addEventListener('DOMContentLoaded', () => {

    // ── Apply section-visibility flags that toggle whole DOM regions ───────
    if (!Config.show.paramsPanel) {
        document.querySelectorAll('.params-mount').forEach(el => el.style.display = 'none');
    }
    if (!Config.show.autoRegenerate) {
        document.querySelectorAll('.auto-toggle').forEach(el => el.style.display = 'none');
    }
    if (!Config.show.downloadButton) {
        document.querySelectorAll('.btn-download').forEach(el => el.style.display = 'none');
    }

    // Apply side titles from config
    document.getElementById('left-title').textContent  = Config.sides.left.title;
    document.getElementById('right-title').textContent = Config.sides.right.title;

    // ── Single-panel view (only view mode; left side stays hidden via .view-ps on <html>/<body>) ──
    function visibleSides() {
        return ['right'];
    }

    // ── State ──────────────────────────────────────────────────────────────
    let realData    = null;
    let realColumns = null;
    let realCsvText = null;

    const panels = { left: new Panel('left'), right: new Panel('right') };
    const paramPanels = {};
    const debounceTimers = {};
    const generationId  = { left: 0, right: 0 };

    // ── LSTM + Lifelong init ───────────────────────────────────────────────
    LSTM.init();
    Lifelong.init();

    // ── Build parameter panels ─────────────────────────────────────────────
    for (const side of ['left', 'right']) {
        const algo   = Config.sides[side].algorithm;
        const schema = Config.algorithms[algo];
        if (!schema) { console.error(`Unknown algorithm for side "${side}": ${algo}`); continue; }
        const mount = document.getElementById(`${side}-params`);
        paramPanels[side] = new ParamPanel(mount, schema, (params) => onParamChange(side, params));
    }

    function onParamChange(side, _params) {
        const auto = document.getElementById(`${side}-auto`).checked;
        if (!auto || !realCsvText) return;
        clearTimeout(debounceTimers[side]);
        debounceTimers[side] = setTimeout(() => runGeneration(side), 500);
    }

    // ── Real-data loader ───────────────────────────────────────────────────
    const realInput   = document.getElementById('real-file');
    const realName    = document.getElementById('real-filename');
    const realInfo    = document.getElementById('real-info');
    const realLoading = document.getElementById('real-loading');
    const realClear   = document.getElementById('real-clear');

    realInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        realName.textContent = file.name;
        realLoading.style.display = 'inline';
        try {
            realCsvText = await file.text();
            await new Promise((resolve, reject) => {
                Papa.parse(realCsvText, {
                    header: true, dynamicTyping: true, skipEmptyLines: true,
                    complete: (res) => {
                        if (!res.data || res.data.length === 0) { reject(new Error('Empty CSV')); return; }
                        realData = res.data;
                        realColumns = (res.meta.fields || []).filter(col =>
                            res.data.some(row => {
                                const v = typeof row[col] === 'string' ? parseFloat(row[col]) : Number(row[col]);
                                return isFinite(v) && !isNaN(v);
                            })
                        );
                        if (realColumns.length === 0) { reject(new Error('No numeric columns')); return; }
                        resolve();
                    },
                    error: (err) => reject(new Error(err.message)),
                });
            });
            realInfo.textContent = `${realData.length} rows · ${realColumns.length} cols`;

            // Show target column popup before starting generation
            const targetCol = await LSTM.requestTargetColumn(realColumns, col => {
                LSTM.setTargetColumn(col);
                Lifelong.setTargetColumn(col);
            });
            LSTM.setTargetColumn(targetCol);
            Lifelong.setTargetColumn(targetCol);

            ['left','right'].forEach(side => {
                document.getElementById(`${side}-generate`).disabled = false;
            });
            await Promise.all(visibleSides().map(side => runGeneration(side)));

        } catch (err) {
            console.error('[real]', err);
            alert(`Error loading real data: ${err.message}`);
            clearReal();
        } finally {
            realLoading.style.display = 'none';
        }
    });

    realClear.addEventListener('click', () => {
        clearReal();
        for (const side of ['left', 'right']) {
            panels[side].clear();
            document.getElementById(`${side}-generate`).disabled = true;
            document.getElementById(`${side}-download`).disabled = true;
        }
    });

    function clearReal() {
        realData = null;
        realColumns = null;
        realCsvText = null;
        realInput.value = '';
        realName.textContent = 'No file selected';
        realInfo.textContent = '';
    }

    // ── Per-side generate / download ───────────────────────────────────────
    ['left', 'right'].forEach(side => {
        document.getElementById(`${side}-generate`).addEventListener('click', () => {
            runGeneration(side);
        });
        document.getElementById(`${side}-download`).addEventListener('click', () => {
            const p = panels[side];
            if (!p.data || !p.columns) return;
            const algo = Config.sides[side].algorithm;
            const csv  = Generator.toCsv(p.columns, p.data);
            const blob = new Blob([csv], { type: 'text/csv' });
            const url  = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `synthetic_${algo}_${side}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        });
    });

    async function runGeneration(side) {
        if (!realCsvText) return;
        const algo   = Config.sides[side].algorithm;
        const params = paramPanels[side].getValues();
        const content = document.getElementById(`${side}-content`);
        const loading = document.getElementById(`${side}-loading`);

        const myId = ++generationId[side];
        loading.style.display = 'inline';

        try {
            const result = await Generator.run(algo, realCsvText, params);
            if (myId !== generationId[side]) return;

            panels[side].setGenerated(result);
            await panels[side].render(realData, realColumns);
            document.getElementById(`${side}-download`).disabled = false;

            // Produce CSV text from generated data and pass to LSTM + Lifelong
            const csvText = Generator.toCsv(result.columns, result.rows);
            LSTM.setSourceCsv(side, csvText);
            Lifelong.setSourceCsv(side, csvText);

        } catch (err) {
            console.error(`[${side}]`, err);
            content.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.purge(el));
            content.innerHTML = `<div class="error-msg"><strong>Generation failed:</strong> ${err.message}</div>`;
            document.getElementById(`${side}-download`).disabled = true;
        } finally {
            if (myId === generationId[side]) loading.style.display = 'none';
        }
    }

    // ── Synchronised scrolling ─────────────────────────────────────────────
    const leftC  = document.getElementById('left-content');
    const rightC = document.getElementById('right-content');
    let syncing  = false;
    leftC.addEventListener('scroll', () => {
        if (syncing) return; syncing = true;
        rightC.scrollTop = leftC.scrollTop;
        syncing = false;
    });
    rightC.addEventListener('scroll', () => {
        if (syncing) return; syncing = true;
        leftC.scrollTop = rightC.scrollTop;
        syncing = false;
    });
});
