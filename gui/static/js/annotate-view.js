/**
 * Annotate View — sequential annotation workflow.
 *
 * Shows blocks that need annotation (stage: detected).
 * For each: shows overlay, user can Accept / Edit (subprocess) / Skip.
 * Handles loading, cancel, error, and timeout states.
 */

document.addEventListener('alpine:init', () => {
  Alpine.data('annotateView', () => ({
    queue: [],
    currentIdx: 0,
    state: 'idle',       // idle | loading | ready | editing | error
    errorMsg: '',
    detecting: false,
    editorMtime: null,
    editorPollId: null,

    get current() {
      return this.queue[this.currentIdx] || null;
    },

    get progress() {
      if (this.queue.length === 0) return '';
      return (this.currentIdx + 1) + ' / ' + this.queue.length;
    },

    async init() {
      // Re-fetch blocks whenever this component initializes
      await this.$store.app.refreshBlocks();
      this.rebuildQueue();

      // Also watch for view changes to refresh queue
      this.$watch('$store.app.view', (view) => {
        if (view === 'annotate') {
          this.$store.app.refreshBlocks().then(() => this.rebuildQueue());
        }
      });
    },

    rebuildQueue() {
      const blocks = this.$store.app.blocks;
      this.queue = blocks.filter(b => b.stage === 'detected');
      if (this.currentIdx >= this.queue.length) {
        this.currentIdx = 0;
      }
    },

    async prepareAndDetect() {
      const block = this.current;
      if (!block) return;

      this.state = 'loading';
      this.detecting = true;

      try {
        await API.post('/api/detection/' + block.name + '/run');
        await this.$store.app.refreshBlocks();
        this.state = 'ready';
      } catch (e) {
        this.state = 'error';
        this.errorMsg = 'Detection failed: ' + e.message;
      } finally {
        this.detecting = false;
      }
    },

    cancelDetection() {
      this.state = 'idle';
      this.detecting = false;
    },

    overlayUrl() {
      return this.current ? '/api/detection/' + this.current.name + '/overlay' : '';
    },

    async accept() {
      const name = this.current.name;
      try {
        await API.post('/api/annotations/' + name, {
          block_name: name,
          status: 'complete',
          source: 'auto-accepted',
        });
      } catch (_) { /* best effort */ }

      await this.$store.app.refreshBlocks();
      this.rebuildQueue();
      this.state = 'idle';
    },

    async launchEditor() {
      const name = this.current.name;
      this.state = 'editing';
      try {
        const res = await API.post('/api/annotations/' + name + '/launch-editor');
        this.editorMtime = res.mtime_before;
        this.startEditorPoll(name);
      } catch (e) {
        this.state = 'ready';
        alert('Failed to launch editor: ' + e.message);
      }
    },

    startEditorPoll(name) {
      this.stopEditorPoll();
      this.editorPollId = setInterval(async () => {
        const mtParam = this.editorMtime != null ? '?mtime_before=' + this.editorMtime : '';
        try {
          const res = await API.get('/api/annotations/' + name + '/editor-status' + mtParam);
          if (res.status === 'saved') {
            this.stopEditorPoll();
            await this.$store.app.refreshBlocks();
            this.rebuildQueue();
            this.state = 'idle';
          } else if (res.status === 'skipped' || res.status === 'not_started') {
            this.stopEditorPoll();
            this.state = 'ready';
          }
        } catch (e) {
          console.error('Editor poll error:', e);
        }
      }, 2000);
    },

    stopEditorPoll() {
      if (this.editorPollId) {
        clearInterval(this.editorPollId);
        this.editorPollId = null;
      }
    },

    skip() {
      this.currentIdx++;
      if (this.currentIdx >= this.queue.length) {
        this.currentIdx = 0;
      }
      this.state = 'idle';
    },

    retry() {
      this.state = 'idle';
      this.errorMsg = '';
    },
  }));
});
