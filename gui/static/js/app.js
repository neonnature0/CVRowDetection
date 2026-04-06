/**
 * Alpine.js global store and hash router.
 */
document.addEventListener('alpine:init', () => {
  Alpine.store('app', {
    view: 'map',          // current view: map | annotate | library | train | verify
    blocks: [],           // cached block list
    loading: false,
    error: null,

    // Help / onboarding state
    showWelcome: false,
    tabHelpDismissed: {},

    async init() {
      // Show welcome on first visit
      if (!localStorage.getItem('cvrd-welcome-seen')) {
        this.showWelcome = true;
      }
      // Load dismissed tab help state
      try {
        const saved = localStorage.getItem('cvrd-tab-help');
        if (saved) this.tabHelpDismissed = JSON.parse(saved);
      } catch (_) {}

      this.route(window.location.hash);
      window.addEventListener('hashchange', () => this.route(window.location.hash));
      await this.refreshBlocks();
    },

    route(hash) {
      const view = (hash || '#map').slice(1);
      const valid = ['map', 'annotate', 'library', 'train', 'verify', 'progress'];
      this.view = valid.includes(view) ? view : 'map';
    },

    toggleWelcome() {
      this.showWelcome = !this.showWelcome;
    },

    dismissWelcome() {
      this.showWelcome = false;
      localStorage.setItem('cvrd-welcome-seen', '1');
    },

    dismissTabHelp(tab) {
      this.tabHelpDismissed[tab] = true;
      localStorage.setItem('cvrd-tab-help', JSON.stringify(this.tabHelpDismissed));
    },

    isTabHelpVisible(tab) {
      return !this.tabHelpDismissed[tab];
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
