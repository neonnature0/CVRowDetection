/**
 * Train View — generate training data, train model, SSE progress.
 */

document.addEventListener('alpine:init', () => {
  Alpine.data('trainView', () => ({
    stats: null,
    state: 'idle',  // idle | generating | training | complete | error
    errorMsg: '',
    progress: null,  // { epoch, total_epochs, train_loss, val_dice, best_dice, status }
    eventSource: null,

    async init() {
      await this.loadStats();
      this.$watch('$store.app.view', (view) => {
        if (view === 'train') this.loadStats();
      });
    },

    async loadStats() {
      try {
        this.stats = await API.get('/api/training/stats');
        // If training is already running, reconnect SSE
        if (this.stats.training_status === 'running') {
          this.state = 'training';
          this.connectSSE();
        }
      } catch (e) {
        console.error('Failed to load training stats:', e);
      }
    },

    async generate() {
      this.state = 'generating';
      this.errorMsg = '';
      try {
        await API.post('/api/training/generate');
        await this.loadStats();
        this.state = 'idle';
      } catch (e) {
        this.state = 'error';
        this.errorMsg = 'Generate failed: ' + e.message;
      }
    },

    async startTraining() {
      this.state = 'training';
      this.progress = null;
      this.errorMsg = '';
      try {
        await API.post('/api/training/start');
        this.connectSSE();
      } catch (e) {
        this.state = 'error';
        this.errorMsg = 'Start training failed: ' + e.message;
      }
    },

    connectSSE() {
      this.disconnectSSE();
      this.eventSource = new EventSource('/api/training/progress');
      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.progress = data;
          if (data.status === 'complete' || data.status === 'stopped' || data.status === 'failed') {
            this.disconnectSSE();
            this.state = data.status === 'complete' ? 'complete' : 'error';
            if (data.status === 'failed') this.errorMsg = data.error || 'Training failed';
            if (data.status === 'stopped') this.errorMsg = 'Training stopped by user';
            this.loadStats();
          }
        } catch (e) {
          console.error('SSE parse error:', e);
        }
      };
      this.eventSource.onerror = () => {
        this.disconnectSSE();
        // Check if training actually finished
        this.loadStats().then(() => {
          if (this.stats && this.stats.training_status !== 'running') {
            this.state = 'complete';
          }
        });
      };
    },

    disconnectSSE() {
      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }
    },

    async stopTraining() {
      try {
        await API.post('/api/training/stop');
      } catch (e) {
        console.error('Stop failed:', e);
      }
    },

    async invalidateAll() {
      if (!confirm('Clear all detection caches? You will need to re-detect all blocks.')) return;
      try {
        await API.post('/api/training/invalidate-all');
        await this.$store.app.refreshBlocks();
        alert('All detection caches cleared. Re-detect blocks to use the new model.');
      } catch (e) {
        alert('Failed: ' + e.message);
      }
    },

    get progressPct() {
      if (!this.progress || !this.progress.total_epochs) return 0;
      return Math.round((this.progress.epoch / this.progress.total_epochs) * 100);
    },

    reset() {
      this.state = 'idle';
      this.errorMsg = '';
      this.progress = null;
    },
  }));
});
