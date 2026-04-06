/**
 * Progress View — pipeline improvement tracking dashboard.
 *
 * Panels: Runs timeline, Learning curve, Block trajectories,
 *         Run comparison, Calibration.
 */

document.addEventListener('alpine:init', () => {
  Alpine.data('progressView', () => ({
    panel: 'runs',
    runs: [],
    expandedRun: null,

    // Learning curve
    learningCurveMsg: '',
    learningChart: null,

    // Block trajectories
    worstBlocks: [],
    trajectoryCharts: {},

    // Regions
    regionCounts: [],
    regionTimeline: [],
    regionEligible: [],
    regionCountsChart: null,
    regionF1Chart: null,

    // Run comparison
    compareOld: '',
    compareNew: '',
    comparing: false,
    comparisonResult: null,
    comparisonError: '',

    get expandedRunData() {
      if (!this.expandedRun) return null;
      return this.runs.find(r => r.run_id === this.expandedRun) || null;
    },

    get warningVisible() {
      if (this.runs.length === 0) return false;
      const latest = this.runs.find(r => r.aggregate_metrics);
      if (!latest) return false;
      return (latest.aggregate_metrics.total_blocks_evaluated || 0) < 15;
    },

    get warningText() {
      const latest = this.runs.find(r => r.aggregate_metrics);
      const n = latest ? (latest.aggregate_metrics.total_blocks_evaluated || 0) : 0;
      return 'Sample size is small (N=' + n + '). F1 differences smaller than approximately 3% are indistinguishable from noise at this sample size. Focus on big changes until you have 15+ blocks.';
    },

    get regionImbalanceWarning() {
      if (this.regionCounts.length <= 1) return '';
      const counts = this.regionCounts.filter(r => r.count > 0);
      if (counts.length <= 1) return '';
      const largest = counts[0]; // sorted descending
      const smallest = counts[counts.length - 1];
      const ratio = largest.count / Math.max(smallest.count, 1);
      if (ratio <= 5) return '';
      return 'Training set is imbalanced: ' + largest.region + ' has ' + largest.count +
        ' blocks, ' + smallest.region + ' has ' + smallest.count +
        ' blocks. The model will be biased toward over-represented regions. ' +
        'Consider annotating more blocks in under-represented regions, or adding sampling weights during training.';
    },

    get comparisonHeadline() {
      if (!this.comparisonResult || !this.comparisonResult.paired_test) return '';
      const t = this.comparisonResult.paired_test;
      const sign = t.mean_diff > 0 ? '+' : '';
      const diff = sign + (t.mean_diff * 100).toFixed(1) + '%';
      const ci = '[' + (t.ci_lower > 0 ? '+' : '') + (t.ci_lower * 100).toFixed(1) + '%, '
               + (t.ci_upper > 0 ? '+' : '') + (t.ci_upper * 100).toFixed(1) + '%]';
      const label = t.significant ? 'SIGNIFICANT' : 'NOT SIGNIFICANT';
      return 'Mean F1 change: ' + diff + ', 95% CI: ' + ci + ', ' + label;
    },

    async init() {
      await this.loadRuns();
    },

    async loadRuns() {
      try {
        this.runs = await API.get('/api/progress/runs');
      } catch (e) {
        console.error('Failed to load runs:', e);
      }
    },

    formatDate(ts) {
      if (!ts) return '—';
      const d = new Date(ts);
      return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    fmtCI(run, metric) {
      const m = run.aggregate_metrics;
      if (!m || m[metric] == null) return '—';
      const val = (m[metric] * 100).toFixed(1) + '%';
      const ci = run.bootstrap_ci_95 && run.bootstrap_ci_95[metric];
      if (ci) {
        return val + ' [' + (ci[0] * 100).toFixed(0) + '–' + (ci[1] * 100).toFixed(0) + ']';
      }
      return val;
    },

    fmtFailures(run) {
      const m = run.aggregate_metrics;
      if (!m || !m.failure_mode_counts) return '—';
      const f = m.failure_mode_counts;
      return (f.false_positives || 0) + '/' + (f.false_negatives || 0);
    },

    // ── Regions ──

    async loadRegionSummary() {
      try {
        const data = await API.get('/api/progress/region-summary');
        this.regionCounts = data.region_counts || [];
        this.regionTimeline = data.region_f1_timeline || [];
        this.regionEligible = data.eligible_regions || [];
        this.$nextTick(() => this.renderRegionCharts());
      } catch (e) {
        console.error('Failed to load region summary:', e);
      }
    },

    renderRegionCharts() {
      // Element A: Region counts bar chart
      if (this.regionCountsChart) { this.regionCountsChart.destroy(); this.regionCountsChart = null; }
      const countsCanvas = document.getElementById('region-counts-chart');
      if (countsCanvas && this.regionCounts.length > 0) {
        const labels = this.regionCounts.map(r => r.region);
        const values = this.regionCounts.map(r => r.count);
        this.regionCountsChart = new Chart(countsCanvas, {
          type: 'bar',
          data: {
            labels,
            datasets: [{
              label: 'Blocks',
              data: values,
              backgroundColor: 'rgba(74,158,255,0.6)',
              borderColor: '#4a9eff',
              borderWidth: 1,
            }],
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
              legend: { display: false },
              annotation: undefined,
            },
            scales: {
              x: {
                ticks: { color: '#888' }, grid: { color: '#333' },
                title: { display: true, text: 'Block count', color: '#888' },
              },
              y: { ticks: { color: '#e0e0e0', font: { size: 11 } }, grid: { display: false } },
            },
          },
        });
      }

      // Element B: Per-region F1 over time
      if (this.regionF1Chart) { this.regionF1Chart.destroy(); this.regionF1Chart = null; }
      const f1Canvas = document.getElementById('region-f1-chart');
      if (f1Canvas && this.regionEligible.length > 0 && this.regionTimeline.length > 0) {
        const labels = this.regionTimeline.map((_, i) => 'R' + (i + 1));
        const colors = ['#4a9eff', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22', '#3498db', '#e91e63', '#00bcd4', '#ff9800'];
        const datasets = this.regionEligible.map((region, idx) => ({
          label: region,
          data: this.regionTimeline.map(t => t.regions[region] != null ? t.regions[region] : null),
          borderColor: colors[idx % colors.length],
          spanGaps: false,  // show gaps, don't interpolate
          pointRadius: 3,
          tension: 0,
        }));

        this.regionF1Chart = new Chart(f1Canvas, {
          type: 'line',
          data: { labels, datasets },
          options: {
            responsive: true,
            plugins: {
              legend: { labels: { color: '#e0e0e0', font: { size: 10 } } },
            },
            scales: {
              x: { ticks: { color: '#888' }, grid: { color: '#333' } },
              y: { min: 0, max: 1, ticks: { color: '#888' }, grid: { color: '#333' }, title: { display: true, text: 'Mean F1 (0.4x)', color: '#888' } },
            },
          },
        });
      }
    },

    // ── Learning Curve ──

    async loadLearningCurve() {
      try {
        const data = await API.get('/api/progress/learning-curve');
        this.renderLearningCurve(data);
      } catch (e) {
        this.learningCurveMsg = 'Failed to load learning curve data.';
      }
    },

    renderLearningCurve(data) {
      if (this.learningChart) {
        this.learningChart.destroy();
        this.learningChart = null;
      }

      const canvas = document.getElementById('learning-curve-chart');
      if (!canvas) return;

      if (data.points.length === 0) {
        this.learningCurveMsg = 'No training runs recorded yet.';
        return;
      }

      const pts = data.points.filter(p => p.mean_f1_04 != null);
      const labels = pts.map(p => p.train_set_size);

      const datasets = [
        { label: 'F1 (0.4x)', data: pts.map(p => p.mean_f1_04), borderColor: '#4a9eff', backgroundColor: 'rgba(74,158,255,0.1)' },
        { label: 'F1 (0.2x)', data: pts.map(p => p.mean_f1_02), borderColor: '#2ecc71', backgroundColor: 'rgba(46,204,113,0.1)' },
        { label: 'F1 (0.1x)', data: pts.map(p => p.mean_f1_01), borderColor: '#f39c12', backgroundColor: 'rgba(243,156,18,0.1)' },
      ];

      // Add fitted curve if available
      if (data.fit) {
        const a = data.fit.a, b = data.fit.b, c = data.fit.c;
        const maxSize = Math.max(...labels) * 1.5;
        const fitX = [];
        const fitY = [];
        for (let x = Math.min(...labels); x <= maxSize; x += 1) {
          fitX.push(x);
          fitY.push(a - b * Math.pow(x, -c));
        }
        datasets.push({
          label: 'Fitted (asymptote=' + (a * 100).toFixed(1) + '%)',
          data: fitY,
          borderColor: 'rgba(74,158,255,0.4)',
          borderDash: [6, 3],
          pointRadius: 0,
          fill: false,
        });
        // Use fitX as labels for the fitted curve dataset
        // Chart.js needs x-axis labels, so we use the actual data x values
      }

      this.learningChart = new Chart(canvas, {
        type: 'line',
        data: { labels, datasets },
        options: {
          responsive: true,
          plugins: {
            legend: { labels: { color: '#e0e0e0', font: { size: 11 } } },
          },
          scales: {
            x: { title: { display: true, text: 'Training set size (blocks)', color: '#888' }, ticks: { color: '#888' }, grid: { color: '#333' } },
            y: { title: { display: true, text: 'Mean F1', color: '#888' }, ticks: { color: '#888' }, grid: { color: '#333' }, min: 0, max: 1 },
          },
        },
      });

      if (data.n_training_runs < 4) {
        this.learningCurveMsg = 'Need at least 4 training runs to fit a learning curve — currently have ' + data.n_training_runs + '.';
      } else if (data.fit) {
        this.learningCurveMsg = 'Predicted ceiling F1: ' + (data.fit.asymptote * 100).toFixed(1) + '%.';
      } else {
        this.learningCurveMsg = '';
      }
    },

    // ── Block Trajectories ──

    async loadTrajectories() {
      try {
        // Get worst blocks from latest run
        const latestRun = this.runs.find(r => r.aggregate_metrics);
        if (!latestRun) {
          this.worstBlocks = [];
          return;
        }

        // Load all block results and find the 5 worst from latest run
        const allResults = await API.get('/api/progress/runs/' + latestRun.run_id);
        // We need per-block data — load from block results
        const blockResults = [];
        // Get block results for this run from the full list
        const response = await fetch('/api/progress/runs');
        // Instead, get the worst block from aggregate
        const worstId = latestRun.aggregate_metrics.worst_block_id;

        // Load trajectories for worst blocks (we know worst from all runs)
        // Get unique block IDs from all runs that have evaluations
        const allBlockData = await fetch('/api/progress/block-trajectory/' + (worstId || ''));

        // Simpler approach: get the 5 worst blocks from per-block results
        this.worstBlocks = [];
        if (!worstId) return;

        // Get all blocks from latest run's per-block data
        // We need a different approach — get all per-block results
        const perBlockRes = await fetch('/api/progress/block-trajectory/' + worstId);
        const trajectory = await perBlockRes.json();

        // For now, show the worst block's trajectory
        if (trajectory.length > 0) {
          this.worstBlocks = [{
            block_id: worstId,
            n_runs: Math.min(trajectory.length, 10),
            data: trajectory.slice(-10),
          }];
          this.$nextTick(() => this.renderTrajectoryCharts());
        }
      } catch (e) {
        console.error('Failed to load trajectories:', e);
      }
    },

    renderTrajectoryCharts() {
      // Destroy old charts
      Object.values(this.trajectoryCharts).forEach(c => c.destroy());
      this.trajectoryCharts = {};

      for (const wb of this.worstBlocks) {
        const canvas = document.getElementById('traj-' + wb.block_id);
        if (!canvas) continue;

        const labels = wb.data.map((_, i) => 'R' + (i + 1));
        const f1Data = wb.data.map(d => d.f1_04);

        // Use spanGaps: false to show broken lines for missing data
        this.trajectoryCharts[wb.block_id] = new Chart(canvas, {
          type: 'line',
          data: {
            labels,
            datasets: [{
              label: 'F1 (0.4x)',
              data: f1Data,
              borderColor: '#4a9eff',
              spanGaps: false,
              pointRadius: 3,
            }],
          },
          options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
              x: { ticks: { color: '#888', font: { size: 9 } }, grid: { color: '#333' } },
              y: { min: 0, max: 1, ticks: { color: '#888', font: { size: 9 } }, grid: { color: '#333' } },
            },
          },
        });
      }
    },

    // ── Run Comparison ──

    async runComparison() {
      if (!this.compareOld || !this.compareNew) return;
      this.comparing = true;
      this.comparisonResult = null;
      this.comparisonError = '';

      try {
        this.comparisonResult = await API.get(
          '/api/progress/compare/' + encodeURIComponent(this.compareOld) + '/' + encodeURIComponent(this.compareNew)
        );
      } catch (e) {
        this.comparisonError = 'Comparison failed: ' + e.message;
      } finally {
        this.comparing = false;
      }
    },
  }));
});
