import { icon } from './icons.js';
import { escapeHtml } from '../core/format.js';

export function toast(message, { tone = 'info', timeout = 3500 } = {}) {
  const region = document.getElementById('toasts');
  if (!region) return;
  const el = document.createElement('div');
  el.className = `toast toast--${tone}`;
  el.innerHTML = `${icon(tone === 'error' ? 'alert' : tone === 'success' ? 'check' : 'info', { size: 16 })}<span>${escapeHtml(message)}</span>`;
  region.appendChild(el);
  requestAnimationFrame(() => el.classList.add('is-visible'));
  setTimeout(() => {
    el.classList.remove('is-visible');
    setTimeout(() => el.remove(), 250);
  }, timeout);
}
