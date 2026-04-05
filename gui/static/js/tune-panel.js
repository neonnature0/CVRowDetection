/**
 * Tune Panel — parameter tuning with before/after overlay comparison.
 *
 * Appears inside the Library view when a detected block is selected.
 * Shows 6 key parameters as sliders/dropdowns, "Run with these params"
 * button, and side-by-side before/after comparison.
 */

document.addEventListener('alpine:init', () => {
  Alpine.data('tunePanel', () => ({
    blockName: null,
    paramDefs: null,    // from /api/detection/tunable-params
    params: {},         // current slider/dropdown values
    state: 'idle',      // idle | running | comparing
    tunedResult: null,  // metrics from tuned run
    errorMsg: '',

    get paramList() {
      // Convert paramDefs object to array for x-for iteration
      if (!this.paramDefs) return [];
      return Object.entries(this.paramDefs).map(([key, def]) => ({ key, ...def }));
    },

    async init() {
      try {
        this.paramDefs = await API.get('/api/detection/tunable-params');
        this.resetToDefaults();
      } catch (e) {
        console.error('Failed to load tunable params:', e);
      }
    },

    resetToDefaults() {
      // Set all params to PipelineConfig defaults
      if (!this.paramDefs) return;
      this.params = {};
      for (const [key, def] of Object.entries(this.paramDefs)) {
        this.params[key] = def.default;
      }
    },

    async resetToSavedConfig() {
      // Reset to saved per-block tuned config (if any), otherwise defaults
      this.resetToDefaults();
      if (!this.blockName) return;
      try {
        const saved = await API.get('/api/detection/' + this.blockName + '/tuned-config');
        if (saved && Object.keys(saved).length > 0) {
          for (const [key, val] of Object.entries(saved)) {
            if (key in this.params) this.params[key] = val;
          }
        }
      } catch (_) {}
    },

    resetToTrueDefaults() {
      // Reset to PipelineConfig dataclass defaults, ignoring any saved config
      this.resetToDefaults();
      this.tunedResult = null;
      this.state = 'idle';
    },

    async loadBlockConfig(name) {
      this.blockName = name;
      this.state = 'idle';
      this.tunedResult = null;
      this.resetToDefaults();

      // Load saved tuned config if it exists
      try {
        const saved = await API.get('/api/detection/' + name + '/tuned-config');
        if (saved && Object.keys(saved).length > 0) {
          for (const [key, val] of Object.entries(saved)) {
            if (key in this.params) this.params[key] = val;
          }
        }
      } catch (_) { /* no saved config */ }
    },

    async runTuned() {
      if (!this.blockName) return;
      this.state = 'running';
      this.errorMsg = '';
      this.tunedResult = null;

      try {
        this.tunedResult = await API.post(
          '/api/detection/' + this.blockName + '/tune',
          { params: { ...this.params } }
        );
        this.state = 'comparing';
      } catch (e) {
        this.state = 'idle';
        this.errorMsg = 'Tuned detection failed: ' + e.message;
      }
    },

    async applyTuned() {
      if (!this.blockName) return;
      try {
        await API.post('/api/detection/' + this.blockName + '/apply-tuned');
        this.state = 'idle';
        // Refresh the library to show new thumbnail
        this.$store.app.refreshBlocks();
      } catch (e) {
        alert('Apply failed: ' + e.message);
      }
    },

    defaultOverlayUrl() {
      return '/api/detection/' + this.blockName + '/overlay';
    },

    tunedOverlayUrl() {
      return '/api/detection/' + this.blockName + '/tuned-overlay?t=' + Date.now();
    },

    isModified(key) {
      return this.paramDefs && this.params[key] !== this.paramDefs[key].default;
    },
  }));
});
