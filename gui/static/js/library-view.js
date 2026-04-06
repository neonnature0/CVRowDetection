/**
 * Library View — block grid with thumbnails, detection actions, annotation launch.
 */

document.addEventListener('alpine:init', () => {
  Alpine.data('libraryView', () => ({
    filter: 'all',  // all | draft | detected | annotated | verified
    expandedBlock: null,
    detecting: null,  // block name currently being detected
    editorBlock: null,  // block name with open editor
    editorMtime: null,
    editorPollId: null,

    get filteredBlocks() {
      const blocks = this.$store.app.blocks;
      if (this.filter === 'all') return blocks;
      return blocks.filter(b => (b.stage || 'draft') === this.filter);
    },

    thumbnailUrl(name) {
      return '/api/detection/' + name + '/thumbnail';
    },

    overlayUrl(name) {
      return '/api/detection/' + name + '/overlay';
    },

    hasDetection(block) {
      return block.last_detection_at != null;
    },

    async runDetection(name) {
      this.detecting = name;
      try {
        await API.post('/api/detection/' + name + '/run?force=true');
        await this.$store.app.refreshBlocks();
      } catch (e) {
        alert('Detection failed: ' + e.message);
      } finally {
        this.detecting = null;
      }
    },

    async launchAnnotate(name) {
      // First ensure detection has run
      const block = this.$store.app.blocks.find(b => b.name === name);
      if (!this.hasDetection(block)) {
        alert('Run detection first before annotating.');
        return;
      }
      try {
        const res = await API.post('/api/annotations/' + name + '/launch-editor');
        this.editorBlock = name;
        this.editorMtime = res.mtime_before;
        this.startEditorPoll(name);
      } catch (e) {
        if (e.message.includes('409')) {
          alert('Editor already open for this block.');
        } else {
          alert('Failed to launch editor: ' + e.message);
        }
      }
    },

    async launchBlindAnnotate(name) {
      // Create blank annotation (0 rows) then open editor.
      // Important: this should be used BEFORE viewing the overlay for this block.
      try {
        await API.post('/api/annotations/' + name + '/prepare-blind');
        const res = await API.post('/api/annotations/' + name + '/launch-editor');
        this.editorBlock = name;
        this.editorMtime = res.mtime_before;
        this.startEditorPoll(name);
      } catch (e) {
        alert('Failed to launch blind editor: ' + e.message);
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
            this.editorBlock = null;
            await this.$store.app.refreshBlocks();
          } else if (res.status === 'skipped' || res.status === 'not_started') {
            this.stopEditorPoll();
            this.editorBlock = null;
          }
        } catch (e) {
          console.error('Poll error:', e);
        }
      }, 2000);
    },

    stopEditorPoll() {
      if (this.editorPollId) {
        clearInterval(this.editorPollId);
        this.editorPollId = null;
      }
    },

    async deleteBlock(name) {
      if (!confirm('Delete block ' + name + '? This removes all associated data.')) return;
      try {
        await API.del('/api/blocks/' + name);
        if (this.expandedBlock === name) this.expandedBlock = null;
        await this.$store.app.refreshBlocks();
      } catch (e) {
        alert('Delete failed: ' + e.message);
      }
    },

    expand(name) {
      this.expandedBlock = this.expandedBlock === name ? null : name;
    },

    difficultyLabels: {
      1: 'Easy — high contrast, bare inter-rows, straight rows, no confounders',
      2: 'Moderate — minor challenges (slight curvature, some grass)',
      3: 'Average — typical real-world block with mixed conditions',
      4: 'Hard — significant challenges (dense inter-row vegetation, curves, shadows)',
      5: 'Very hard — multiple confounders (dirt paths, trees, sun angle issues, L-shapes)',
    },

    async setDifficulty(name, rating) {
      try {
        await API.patch('/api/blocks/' + name + '/difficulty', { difficulty_rating: rating });
        await this.$store.app.refreshBlocks();
      } catch (e) {
        console.error('Failed to set difficulty:', e);
      }
    },
  }));
});
