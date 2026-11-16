// Charlie AI: the existing /api/chat assistant with a refined conversation UI,
// voice input, read-aloud, and station/line chips detected in replies.

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { LINES } from '../core/constants.js';
import { escapeHtml as e, renderRichText, formatClockSeconds } from '../core/format.js';
import { icon } from '../ui/icons.js';
import { lineBadge, arrivalRow } from '../ui/components.js';
import { openStation } from '../ui/station-panel.js';
import { requestLocation } from '../ui/location.js';

const SUGGESTIONS = [
  'When is the next Red Line train at Harvard?',
  'Are there delays on the Orange Line?',
  'How do I get from Fenway to Park Street?',
  'Show me service alerts',
];

let recognition = null;
let listening = false;
let sending = false;

/** Stations and subway lines mentioned in a reply, for quick-action chips. */
export function mentions(text, stations) {
  const lower = ` ${String(text).toLowerCase()} `;
  const found = [];
  const sorted = [...stations].sort((a, b) => b.name.length - a.name.length);
  let scrub = lower;
  for (const s of sorted) {
    const n = s.name.toLowerCase();
    if (n.length < 4) continue;
    const re = new RegExp(`[^a-z]${n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}[^a-z]`);
    if (re.test(scrub)) { found.push({ type: 'station', id: s.id, name: s.name }); scrub = scrub.replace(re, ' '); }
    if (found.length >= 4) break;
  }
  const lines = [];
  for (const [id, l] of Object.entries(LINES)) {
    if (id.startsWith('Green-')) continue;
    if (lower.includes(l.name.toLowerCase())) lines.push({ type: 'line', id, name: l.name });
  }
  if (/green line/.test(lower) && !lines.some(l => l.id.startsWith('Green'))) lines.push({ type: 'line', id: 'Green-B', name: 'Green Line' });
  return [...found, ...lines.slice(0, 3)];
}

/** Structured arrivals under a reply: the same predictions, with a station-board shortcut. */
export function arrivalsCardHtml(a) {
  if (!a) return '';
  const station = a.station;
  const open = station ? `<button class="btn btn--secondary btn--sm" data-open-station="${e(station.id)}">${icon('pin', { size: 14 })}Open ${e(station.name)} station board</button>` : '';
  if (a.status === 'ambiguous') {
    const line = a.route ? ` ${a.route.length <= 3 && /\d/.test(a.route) ? `route ${a.route}` : a.route}` : '';
    return `<div class="msg-card"><div class="chips">${(a.candidates || []).map(c =>
      `<button class="chip chip--btn" data-suggest="${e(`Next${line} train at ${c}`)}">${icon('pin', { size: 14 })}${e(c)}</button>`).join('')}</div></div>`;
  }
  if (a.status === 'unknown') return `<div class="msg-card"><a class="btn btn--secondary btn--sm" href="#/stations">${icon('search', { size: 14 })}Browse stations</a></div>`;
  if (a.status === 'unavailable') return `<div class="msg-card msg-card--warn"><p class="small">Live predictions are unavailable right now, so no times are shown.</p><div class="btn-row">${open}</div></div>`;
  if (a.status === 'none') return `<div class="msg-card"><div class="btn-row">${open}<a class="btn btn--ghost btn--sm" href="#/alerts">${icon('alert', { size: 14 })}Service alerts</a></div></div>`;
  return `<div class="msg-card" aria-label="Live arrivals">
    ${(a.groups || []).map(g => `<section class="board board--compact">
      <header class="board__head">${lineBadge(g.route_id, { size: 'sm' })}<span class="board__dir">${e(g.route_name ? `${g.route_name} · ` : '')}To ${e(g.label)}</span></header>
      <ul class="arrivals">${g.items.map(p => arrivalRow(p, { stationName: station?.name })).join('')}</ul>
    </section>`).join('')}
    <div class="msg-card__foot"><span class="muted small">MBTA predictions · checked ${e(formatClockSeconds(a.updated_at))}</span>${open}</div>
  </div>`;
}

function messageHtml(msg, i) {
  if (msg.role === 'user') return `<li class="msg msg--user"><div class="msg__bubble">${e(msg.text)}</div></li>`;
  if (msg.role === 'error') return `<li class="msg msg--bot msg--error"><div class="msg__avatar" aria-hidden="true">${icon('alert', { size: 16 })}</div><div class="msg__bubble">${e(msg.text)} <button class="link" data-retry="${i}">Try again</button></div></li>`;
  const chips = mentions(msg.text, store.get('stations'));
  return `<li class="msg msg--bot">
    <div class="msg__avatar" aria-hidden="true">${icon('sparkle', { size: 16 })}</div>
    <div class="msg__content">
      <div class="msg__bubble"><p>${renderRichText(msg.arrivals?.status === 'ok' ? String(msg.text).split('\n')[0] : msg.text)}</p></div>
      ${arrivalsCardHtml(msg.arrivals)}
      <div class="msg__actions">
        ${(msg.arrivals ? chips.filter(c => c.type !== 'station') : chips).map(c => c.type === 'station'
          ? `<button class="chip chip--btn" data-open-station="${e(c.id)}">${icon('pin', { size: 14 })}${e(c.name)}</button>`
          : `<a class="chip chip--btn" href="#/lines/${encodeURIComponent(c.id)}">${lineBadge(c.id, { size: 'sm', label: false })}${e(c.name)}</a>`).join('')}
        ${'speechSynthesis' in window ? `<button class="icon-btn icon-btn--sm" data-speak="${i}" aria-label="Read this reply aloud" title="Read aloud">${icon('speaker', { size: 16 })}</button>` : ''}
      </div>
    </div>
  </li>`;
}

function render(el) {
  const chat = store.get('chat');
  const list = el.querySelector('[data-messages]');
  const intro = el.querySelector('[data-intro]');
  intro.hidden = chat.messages.length > 0;
  list.innerHTML = chat.messages.map(messageHtml).join('') +
    (sending ? `<li class="msg msg--bot"><div class="msg__avatar" aria-hidden="true">${icon('sparkle', { size: 16 })}</div><div class="msg__bubble msg__typing" aria-label="Charlie is typing"><span></span><span></span><span></span></div></li>` : '');
  const scroller = el.querySelector('[data-scroll]');
  scroller.scrollTop = scroller.scrollHeight;
}

async function send(el, text) {
  const message = text.trim();
  if (!message || sending) return;
  const chat = store.get('chat');
  chat.messages.push({ role: 'user', text: message });
  sending = true;
  render(el);
  const loc = store.get('location');
  const { data, error } = await api.chat({ message, history: chat.history, location: loc || null, context: chat.context || null });
  sending = false;
  if (error || !data) {
    chat.messages.push({ role: 'error', text: 'Charlie couldn’t reach the transit service just now.', retry: message });
  } else {
    if (Array.isArray(data.history)) chat.history = data.history;
    if (data.context) chat.context = data.context;
    chat.messages.push({ role: 'bot', text: data.reply || 'No reply.', arrivals: data.arrivals || null });
    if (data.request_location) setTimeout(() => requestLocation(), 600);
  }
  render(el);
  el.querySelector('#chat-input')?.focus();
}

function speak(text) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(String(text).replace(/[*_#•]/g, '').replace(/\s+/g, ' '));
  u.rate = 1.05;
  u.lang = 'en-US';
  window.speechSynthesis.speak(u);
}

function setupVoice(el) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = el.querySelector('[data-mic]');
  if (!SR) { btn.hidden = true; return; }
  btn.addEventListener('click', () => {
    if (listening) { recognition?.stop(); return; }
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    const input = el.querySelector('#chat-input');
    recognition.onstart = () => { listening = true; btn.classList.add('is-active'); btn.setAttribute('aria-pressed', 'true'); input.placeholder = 'Listening…'; };
    recognition.onresult = ev => {
      let text = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) text += ev.results[i][0].transcript;
      input.value = text;
      if (ev.results[ev.results.length - 1].isFinal) send(el, text).then(() => { input.value = ''; });
    };
    recognition.onerror = ev => { el.querySelector('[data-voice-status]').textContent = ev.error === 'not-allowed' ? 'Microphone access was denied.' : 'Didn’t catch that — try again.'; };
    recognition.onend = () => { listening = false; btn.classList.remove('is-active'); btn.setAttribute('aria-pressed', 'false'); input.placeholder = 'Ask about trains, buses, delays…'; };
    recognition.start();
  });
}

export const assistantView = {
  enter(el) {
    el.innerHTML = `
      <div class="chat">
        <div class="chat__scroll" data-scroll>
          <div class="chat__inner">
            <header class="chat__intro" data-intro>
              <div class="chat__mark" aria-hidden="true">${icon('sparkle', { size: 24 })}</div>
              <h1>Ask Charlie</h1>
              <p class="muted">Live arrivals, delays and directions in plain English. Charlie uses real-time MBTA data${store.get('location')?.station ? ` and knows you’re near <strong>${e(store.get('location').station)}</strong>` : ''}.</p>
              <div class="suggestions">${SUGGESTIONS.map(s => `<button class="suggestion" data-suggest="${e(s)}">${e(s)}</button>`).join('')}</div>
            </header>
            <ol class="messages" data-messages aria-live="polite" aria-label="Conversation"></ol>
          </div>
        </div>
        <form class="composer" data-composer>
          <button type="button" class="icon-btn" data-loc aria-label="Share my location with Charlie" title="Share my location">${icon('locate', { size: 18 })}</button>
          <label class="sr-only" for="chat-input">Message Charlie</label>
          <input id="chat-input" class="composer__input" placeholder="Ask about trains, buses, delays…" autocomplete="off" />
          <button type="button" class="icon-btn" data-mic aria-label="Voice input" aria-pressed="false" title="Voice input">${icon('mic', { size: 18 })}</button>
          <button type="submit" class="btn btn--primary btn--icon" aria-label="Send">${icon('send', { size: 18 })}</button>
          <span class="sr-only" role="status" data-voice-status></span>
        </form>
      </div>`;
    const input = el.querySelector('#chat-input');
    el.querySelector('[data-composer]').addEventListener('submit', ev => {
      ev.preventDefault();
      const text = input.value;
      input.value = '';
      send(el, text);
    });
    el.addEventListener('click', ev => {
      const t = ev.target.closest('[data-suggest],[data-open-station],[data-speak],[data-retry]');
      if (!t) return;
      if (t.dataset.suggest) send(el, t.dataset.suggest);
      if (t.dataset.openStation) openStation(t.dataset.openStation);
      if (t.dataset.speak) speak(store.get('chat').messages[Number(t.dataset.speak)].text);
      if (t.dataset.retry) {
        const chat = store.get('chat');
        const msg = chat.messages[Number(t.dataset.retry)];
        chat.messages.splice(Number(t.dataset.retry), 1);
        const lastUser = chat.messages.length - 1;
        if (chat.messages[lastUser]?.role === 'user') chat.messages.splice(lastUser, 1);
        send(el, msg.retry);
      }
    });
    el.querySelector('[data-loc]').addEventListener('click', () => requestLocation());
    setupVoice(el);
    render(el);
    input.focus({ preventScroll: true });
  },
  leave() { recognition?.stop(); window.speechSynthesis?.cancel(); },
};
