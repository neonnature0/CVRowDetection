/**
 * Alpine.js global store and hash router.
 */
document.addEventListener('alpine:init', () => {
  Alpine.store('app', {
    view: 'map',          // current view: map | annotate | library | train | verify
    blocks: [],           // cached block list
    loading: false,
    error: null,

    async init() {
      this.route(window.location.hash);
      window.addEventListener('hashchange', () => this.route(window.location.hash));
      await this.refreshBlocks();
    },

    route(hash) {
      const view = (hash || '#map').slice(1);
      const valid = ['map', 'annotate', 'library', 'train', 'verify'];
      this.view = valid.includes(view) ? view : 'map';
    },

    async refreshBlocks() {
      try {
        this.blocks = await API.get('/api/blocks');
      } catch (e) {
        console.error('Failed to load blocks:', e);
        this.error = 'Failed to load blocks';
      }
    },
  });
});
