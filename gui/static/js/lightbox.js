/**
 * Lightbox — full-screen image viewer with zoom and pan.
 *
 * The image renders at its natural resolution. On open, it's scaled
 * to fit the viewport. User can zoom with scroll wheel and pan by
 * dragging. The transform uses translate + scale on an absolutely
 * positioned element, so the image is never clipped by overflow.
 */

document.addEventListener('alpine:init', () => {
  Alpine.store('lightbox', {
    visible: false,
    src: '',
    scale: 1,
    panX: 0,
    panY: 0,
    _dragging: false,
    _lastX: 0,
    _lastY: 0,
    _imgW: 0,
    _imgH: 0,

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

    fitToScreen(img) {
      // Called when the image loads — fit it to the canvas area
      this._imgW = img.naturalWidth;
      this._imgH = img.naturalHeight;
      const canvas = document.querySelector('.lightbox-canvas');
      if (!canvas) return;
      const cw = canvas.clientWidth;
      const ch = canvas.clientHeight;
      // Scale to fit with some padding
      this.scale = Math.min(cw / this._imgW, ch / this._imgH) * 0.95;
      // Center: position absolute element at center of canvas, offset by half image size
      this.panX = (cw / 2) - (this._imgW * this.scale / 2);
      this.panY = (ch / 2) - (this._imgH * this.scale / 2);
    },

    zoom(delta) {
      const factor = 1 + delta * 0.001;
      const newScale = Math.max(0.1, Math.min(20, this.scale * factor));
      // Zoom toward center of canvas
      const canvas = document.querySelector('.lightbox-canvas');
      if (canvas) {
        const cx = canvas.clientWidth / 2;
        const cy = canvas.clientHeight / 2;
        this.panX = cx - (cx - this.panX) * (newScale / this.scale);
        this.panY = cy - (cy - this.panY) * (newScale / this.scale);
      }
      this.scale = newScale;
    },

    zoomAtPoint(delta, clientX, clientY) {
      const factor = 1 + delta * 0.001;
      const newScale = Math.max(0.1, Math.min(20, this.scale * factor));
      const canvas = document.querySelector('.lightbox-canvas');
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const mx = clientX - rect.left;
        const my = clientY - rect.top;
        // Zoom toward mouse position
        this.panX = mx - (mx - this.panX) * (newScale / this.scale);
        this.panY = my - (my - this.panY) * (newScale / this.scale);
      }
      this.scale = newScale;
    },

    resetView() {
      // Re-fit to screen
      const img = document.querySelector('.lightbox-content img');
      if (img && img.naturalWidth) this.fitToScreen(img);
    },
  });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (typeof Alpine === 'undefined') return;
  const lb = Alpine.store('lightbox');
  if (!lb || !lb.visible) return;
  if (e.key === 'Escape') lb.close();
  if (e.key === '0') lb.resetView();
  if (e.key === '+' || e.key === '=') lb.zoom(300);
  if (e.key === '-') lb.zoom(-300);
});

// Mouse handlers (called from HTML event attributes)
function lightboxWheel(e) {
  Alpine.store('lightbox').zoomAtPoint(-e.deltaY, e.clientX, e.clientY);
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
