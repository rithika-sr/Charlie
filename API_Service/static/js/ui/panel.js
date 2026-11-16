// Contextual detail panel: a right-hand drawer on desktop, a draggable bottom
// sheet on mobile. Non-modal; Esc or the close button dismisses it.

import { icon } from './icons.js';

const root = () => document.getElementById('detail-panel');
let onCloseFn = null;
let returnFocus = null;
let currentKey = null;

const isMobile = () => window.matchMedia('(max-width: 767px)').matches;

export function openPanel({ key, title, eyebrow = '', html, onClose = null }) {
  const el = root();
  if (onCloseFn && currentKey !== key) { const fn = onCloseFn; onCloseFn = null; fn(); }
  currentKey = key;
  onCloseFn = onClose;
  if (!el.classList.contains('is-open')) returnFocus = document.activeElement;
  el.querySelector('.panel__eyebrow').textContent = eyebrow;
  el.querySelector('.panel__title').textContent = title;
  el.querySelector('.panel__body').innerHTML = html;
  el.hidden = false;
  el.setAttribute('aria-label', title);
  requestAnimationFrame(() => {
    if (currentKey !== key) return; // closed before the first frame
    el.classList.add('is-open');
    el.dataset.snap = isMobile() ? 'half' : '';
    document.body.classList.add('panel-open');
  });
  el.querySelector('.panel__title').focus({ preventScroll: true });
  return el.querySelector('.panel__body');
}

export function updatePanelBody(key, html) {
  if (currentKey !== key) return null;
  const body = root().querySelector('.panel__body');
  body.innerHTML = html;
  return body;
}

export function panelBody(key) {
  return currentKey === key ? root().querySelector('.panel__body') : null;
}

// Keyed on `currentKey`, which is set synchronously in openPanel. (The `is-open`
// class is added a frame later for the transition, so a fast response must not
// depend on it.)
export function isPanelOpen(key) {
  return currentKey === key;
}

export function closePanel() {
  const el = root();
  if (!el.classList.contains('is-open') && currentKey === null) return;
  el.classList.remove('is-open');
  document.body.classList.remove('panel-open');
  const fn = onCloseFn; onCloseFn = null; currentKey = null;
  if (fn) fn();
  setTimeout(() => { if (!el.classList.contains('is-open')) el.hidden = true; }, 220);
  if (returnFocus && document.contains(returnFocus)) returnFocus.focus({ preventScroll: true });
}

export function initPanel() {
  const el = root();
  el.innerHTML = `
    <div class="panel__handle" aria-hidden="true"><span></span></div>
    <header class="panel__head">
      <div class="panel__titles"><div class="panel__eyebrow"></div><h2 class="panel__title" tabindex="-1"></h2></div>
      <button class="icon-btn" data-panel-close aria-label="Close details">${icon('close')}</button>
    </header>
    <div class="panel__body"></div>`;
  el.querySelector('[data-panel-close]').addEventListener('click', closePanel);
  document.addEventListener('keydown', ev => {
    if (ev.key === 'Escape' && el.classList.contains('is-open') && !ev.defaultPrevented) closePanel();
  });

  // Bottom-sheet drag (mobile): drag the handle/header between half and full, or down to close.
  let startY = null, startSnap = 'half';
  const grab = el.querySelector('.panel__handle');
  const head = el.querySelector('.panel__head');
  const onDown = ev => {
    if (!isMobile() || ev.target.closest('button')) return;
    startY = ev.clientY; startSnap = el.dataset.snap || 'half';
    el.classList.add('is-dragging');
    ev.currentTarget.setPointerCapture?.(ev.pointerId);
  };
  const onMove = ev => {
    if (startY === null) return;
    const dy = ev.clientY - startY;
    el.style.setProperty('--drag', `${Math.max(dy, startSnap === 'full' ? -0 : -window.innerHeight * 0.4)}px`);
  };
  const onUp = ev => {
    if (startY === null) return;
    const dy = ev.clientY - startY;
    startY = null;
    el.classList.remove('is-dragging');
    el.style.removeProperty('--drag');
    if (dy > 120 && startSnap === 'half') return closePanel();
    if (dy > 80 && startSnap === 'full') { el.dataset.snap = 'half'; return; }
    if (dy < -60) { el.dataset.snap = 'full'; return; }
    if (Math.abs(dy) < 6) el.dataset.snap = startSnap === 'half' ? 'full' : 'half';
  };
  for (const t of [grab, head]) {
    t.addEventListener('pointerdown', onDown);
    t.addEventListener('pointermove', onMove);
    t.addEventListener('pointerup', onUp);
    t.addEventListener('pointercancel', onUp);
  }
}
