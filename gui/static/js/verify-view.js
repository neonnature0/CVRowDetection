/**
 * Verify View — batch visual verification of N random blocks.
 */

document.addEventListener('alpine:init', () => {
  Alpine.data('verifyView', () => ({
    n: 10,
    state: 'idle',  // idle | running | done
    progress: { done: 0, total: 0, current: '' },
    results: [],
    eventSource: null,

    get progressPct() {
      if (!this.progress.total) return 0;
      return Math.round((this.progress.done / this.progress.total) * 100);
    },

    async startVerify() {
      this.state = 'running';
      this.results = [];
      this.progress = { done: 0, total: 0, current: '' };

      try {
        const res = await API.post('/api/verify/run?n=' + this.n);
        this.progress.total = res.total;
        this.connectSSE();
      } catch (e) {
        this.state = 'idle';
        alert('Failed to start verification: ' + e.message);
      }
    },

    connectSSE() {
      if (this.eventSource) this.eventSource.close();
      this.eventSource = new EventSource('/api/verify/progress');
      this.eventSource.onmessage = async (event) => {
        try {
          const data = JSON.parse(event.data);
          this.progress = data;
          if (data.status === 'complete') {
            this.eventSource.close();
            this.eventSource = null;
            this.results = await API.get('/api/verify/results');
            this.state = 'done';
            await this.$store.app.refreshBlocks();
          }
        } catch (e) {
          console.error('SSE parse error:', e);
        }
      };
      this.eventSource.onerror = () => {
        if (this.eventSource) this.eventSource.close();
        this.eventSource = null;
        // Try to fetch results anyway
        API.get('/api/verify/results').then(r => {
          this.results = r;
          if (r.length > 0) this.state = 'done';
          else this.state = 'idle';
        });
      };
    },
  }));
});
