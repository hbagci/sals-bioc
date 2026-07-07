class ParamPanel {

    constructor(mountEl, schema, onChange) {
        this.mountEl  = mountEl;
        this.schema   = schema;
        this.onChange = onChange;
        this.values   = {};
        schema.params.forEach(f => { this.values[f.key] = f.default; });
        this._render();
    }

    /** Current parameter values (ready to ship to the backend). */
    getValues() {
        const out = {};
        for (const f of this.schema.params) {
            let v = this.values[f.key];
            if (f.type === 'int' || f.type === 'float') {
                if (v === '' || v === null || v === undefined) v = null;
                else v = f.type === 'int' ? parseInt(v, 10) : parseFloat(v);
            }
            out[f.key] = v;
        }
        return out;
    }

    /** Re-render (e.g. after showIf dependencies changed). */
    _render() {
        const s = this.schema;
        this.mountEl.innerHTML = '';

        const header = document.createElement('div');
        header.className = 'params-header';
        header.innerHTML = `<strong>${s.label}</strong>`;
        if (s.description) {
            const d = document.createElement('div');
            d.className = 'params-desc';
            d.textContent = s.description;
            header.appendChild(d);
        }
        this.mountEl.appendChild(header);

        const grid = document.createElement('div');
        grid.className = 'params-grid';
        this.mountEl.appendChild(grid);

        for (const field of s.params) {
            if (field.visible === false) continue;
            if (field.showIf && !field.showIf(this.values)) continue;
            grid.appendChild(this._renderField(field));
        }
    }

    _renderField(f) {
        const wrap  = document.createElement('label');
        wrap.className = 'param-field';

        const lbl = document.createElement('span');
        lbl.className = 'param-label';
        lbl.textContent = f.label;
        wrap.appendChild(lbl);

        let input;
        if (f.type === 'bool') {
            input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = !!this.values[f.key];
            input.className = 'param-input bool';
            input.addEventListener('change', () => this._set(f.key, input.checked));
        } else if (f.type === 'select') {
            input = document.createElement('select');
            input.className = 'param-input';
            for (const opt of f.options) {
                const o = document.createElement('option');
                o.value = opt; o.textContent = opt;
                if (opt === this.values[f.key]) o.selected = true;
                input.appendChild(o);
            }
            input.addEventListener('change', () => this._set(f.key, input.value));
        } else { // int / float / text
            input = document.createElement('input');
            input.type = 'number';
            if (f.min !== undefined) input.min = f.min;
            if (f.max !== undefined) input.max = f.max;
            if (f.step !== undefined) input.step = f.step;
            input.value = this.values[f.key] === null ? '' : this.values[f.key];
            input.className = 'param-input num';
            input.addEventListener('input', () => this._set(f.key, input.value));
        }

        wrap.appendChild(input);

        if (f.hint) {
            const h = document.createElement('span');
            h.className = 'param-hint';
            h.textContent = f.hint;
            wrap.appendChild(h);
        }
        return wrap;
    }

    _visibleKeys() {
        return this.schema.params
            .filter(f => f.visible !== false && (!f.showIf || f.showIf(this.values)))
            .map(f => f.key)
            .join('|');
    }

    _set(key, value) {
        const before = this._visibleKeys();
        this.values[key] = value;
        const after = this._visibleKeys();
        // Only re-render when a showIf actually flipped — otherwise we'd lose
        // focus on every keystroke.
        if (before !== after) this._render();
        this.onChange(this.getValues());
    }
}
