const Generator = {

    async run(algorithm, csvText, params) {
        const endpoint = Config.api.baseUrl + Config.api.endpoints[algorithm];
        if (!endpoint) throw new Error(`Unknown algorithm: ${algorithm}`);

        const resp = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv: csvText, params }),
        });

        if (!resp.ok) {
            let msg = `HTTP ${resp.status}`;
            try { const j = await resp.json(); if (j.error) msg = j.error; } catch (_) {}
            throw new Error(msg);
        }

        const json = await resp.json();
        // Convert rows-as-arrays into row-objects keyed by column name
        const rows = json.rows.map(r => {
            const o = {};
            json.columns.forEach((c, i) => { o[c] = r[i]; });
            return o;
        });
        return { columns: json.columns, rows, stats: json.stats || {} };
    },

    async lifelong(endpoint, body) {
        const resp = await fetch(Config.api.baseUrl + endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            let msg = `HTTP ${resp.status}`;
            try { const j = await resp.json(); if (j.error) msg = j.error; } catch (_) {}
            throw new Error(msg);
        }
        return resp.json();
    },

    async lstm(endpoint, body) {
        const resp = await fetch(Config.api.baseUrl + endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            let msg = `HTTP ${resp.status}`;
            try { const j = await resp.json(); if (j.error) msg = j.error; } catch (_) {}
            throw new Error(msg);
        }
        return resp.json();
    },

    /** Convert an array of row-objects + column list to CSV text for download. */
    toCsv(columns, rows) {
        const lines = [columns.join(',')];
        for (const r of rows) {
            lines.push(columns.map(c => {
                const v = r[c];
                if (v === null || v === undefined) return '';
                return String(v);
            }).join(','));
        }
        return lines.join('\n');
    },
};
