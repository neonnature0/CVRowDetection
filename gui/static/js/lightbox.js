/**
 * Lightbox — full-screen image viewer with zoom and pan.
 *
 * Usage: Alpine store 'lightbox'
 *   $store.lightbox.open('/api/detection/abc123/overlay')
 *   $store.lightbox.close()
 */

document.addEventListener('alpine:init', () => {
  Alpine.store('lightbox', {
    visible: false,
    src: '',
    // Transform state
    scale: 1,
    panX: 0,
    panY: 0,
    _dragging: false,
    _lastX: 0,
    _lastY: 0,

    open(src) {
      this.src = src;
      this.scale = 1;
      this.panX = 0;
      this.panY = 0;
      this.visible = true;
    },

    close() {
      this.visible = false;
      this.src = '';
    },

    zoom(delta) {
      const prev = this.scale;
      this.scale = Math.max(0.2, Math.min(10, this.scale * (1 + delta * 0.001)));
    },

    resetView() {
      this.scale = 1;
      this.panX = 0;
      this.panY = 0;
    },
  });
});

// Wire up mouse events after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  document.addEventListener('keydown', (e) => {
    const lb = Alpine.store('lightbox');
    if (!lb.visible) return;
    if (e.key === 'Escape') lb.close();
    if (e.key === '0') lb.resetView();
    if (e.key === '+' || e.key === '=') lb.zoom(200);
    if (e.key === '-') lb.zoom(-200);
  });
});

/**
 * Alpine directive-style event handlers for the lightbox container.
 * Called from inline event attributes in the HTML.
 */
function lightboxWheel(e) {
  e.preventDefault();
  Alpine.store('lightbox').zoom(-e.deltaY);
}

function lightboxMouseDown(e) {
  if (e.button !== 0) return;
  const lb = Alpine.store('lightbox');
  lb._dragging = true;
  lb._lastX = e.clientX;
  lb._lastY = e.clientY;
  e.currentTarget.style.cursor = 'grabbing';
}

function lightboxMouseMove(e) {
  const lb = Alpine.store('lightbox');
  if (!lb._dragging) return;
  lb.panX += e.clientX - lb._lastX;
  lb.panY += e.clientY - lb._lastY;
  lb._lastX = e.clientX;
  lb._lastY = e.clientY;
}

function lightboxMouseUp(e) {
  const lb = Alpine.store('lightbox');
  lb._dragging = false;
  if (e.currentTarget) e.currentTarget.style.cursor = 'grab';
}
