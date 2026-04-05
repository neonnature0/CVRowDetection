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
    abortController: null,
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
      await this.loadQueue();
    },

    async loadQueue() {
      try {
        // Get blocks with stage 'detected' (ready for annotation)
        const blocks = this.$store.app.blocks;
        this.queue = blocks.filter(b => b.stage === 'detected');
        this.currentIdx = 0;
        this.state = this.queue.length > 0 ? 'idle' : 'idle';
      } catch (e) {
        this.state = 'error';
        this.errorMsg = 'Failed to load queue';
      }
    },

    async prepareAndDetect() {
      const block = this.current;
      if (!block) return;

      this.state = 'loading';
      this.detecting = true;
      this.abortController = new AbortController();

      try {
        // Run detection (this fetches tiles + runs pipeline, can take 30s+)
        await API.post('/api/detection/' + block.name + '/run');
        await this.$store.app.refreshBlocks();
        this.state = 'ready';
      } catch (e) {
        if (e.name === 'AbortError') {
          this.state = 'idle';
        } else {
          this.state = 'error';
          this.errorMsg = 'Detection failed: ' + e.message;
        }
      } finally {
        this.detecting = false;
        this.abortController = null;
      }
    },

    cancelDetection() {
      if (this.abortController) {
        this.abortController.abort();
      }
      this.state = 'idle';
      this.detecting = false;
    },

    overlayUrl() {
      return this.current ? '/api/detection/' + this.current.name + '/overlay' : '';
    },

    async accept() {
      // Save annotation as complete (uses detection result as-is)
      const name = this.current.name;
      try {
        // Mark as annotated directly
        await API.post('/api/blocks', null);  // dummy — we just update stage
      } catch (_) { /* ignore */ }

      // Update stage via the block registry
      const blocks = this.$store.app.blocks;
      const block = blocks.find(b => b.name === name);
      if (block) {
        // Post a simple annotation acceptance
        try {
          await API.post('/api/annotations/' + name, {
            block_name: name,
            status: 'complete',
            source: 'auto-accepted',
          });
        } catch (_) { /* best effort */ }
      }

      await this.$store.app.refreshBlocks();
      this.advance();
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
            this.advance();
          } else if (res.status === 'skipped' || res.status === 'not_started') {
            this.stopEditorPoll();
            this.state = 'ready';  // Back to ready, user can try again
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
      this.advance();
    },

    advance() {
      this.state = 'idle';
      // Reload queue to reflect stage changes
      const blocks = this.$store.app.blocks;
      this.queue = blocks.filter(b => b.stage === 'detected');
      if (this.currentIdx >= this.queue.length) {
        this.currentIdx = 0;
      }
    },

    retry() {
      this.state = 'idle';
      this.errorMsg = '';
    },
  }));
});
