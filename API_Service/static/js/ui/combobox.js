// Accessible autocomplete (WAI-ARIA combobox with a listbox popup).
// Keyboard: ↓/↑ move, Enter selects, Esc closes, Tab leaves.

let uid = 0;

export function combobox(input, { source, render, onSelect, emptyText = 'No matches', minChars = 1, onOpenEmpty = null }) {
  const id = `cb-${++uid}`;
  const wrapper = input.closest('.combo') || input.parentElement;
  const list = document.createElement('ul');
  list.className = 'combo__list';
  list.id = `${id}-list`;
  list.setAttribute('role', 'listbox');
  list.hidden = true;
  wrapper.appendChild(list);

  const status = document.createElement('div');
  status.className = 'sr-only';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  wrapper.appendChild(status);

  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-expanded', 'false');
  input.setAttribute('aria-controls', list.id);
  input.setAttribute('autocomplete', 'off');

  let items = [];
  let active = -1;

  function close() {
    list.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
    active = -1;
  }

  function setActive(i) {
    active = i;
    [...list.children].forEach((li, idx) => li.setAttribute('aria-selected', String(idx === i)));
    if (i >= 0) {
      input.setAttribute('aria-activedescendant', `${id}-opt-${i}`);
      list.children[i]?.scrollIntoView({ block: 'nearest' });
    } else input.removeAttribute('aria-activedescendant');
  }

  async function update() {
    const q = input.value.trim();
    if (q.length < minChars && !onOpenEmpty) { close(); return; }
    items = q.length < minChars ? await onOpenEmpty() : await source(q);
    list.classList.toggle('combo__list--empty', !items || !items.length);
    if (!items || !items.length) {
      if (q.length < minChars) { close(); return; }
      list.innerHTML = `<li class="combo__empty" role="presentation">${emptyText}</li>`;
      status.textContent = emptyText;
    } else {
      list.innerHTML = items.map((item, i) =>
        `<li id="${id}-opt-${i}" role="option" aria-selected="false" class="combo__opt" data-index="${i}">${render(item)}</li>`).join('');
      status.textContent = `${items.length} result${items.length === 1 ? '' : 's'} available`;
    }
    list.hidden = false;
    input.setAttribute('aria-expanded', 'true');
    setActive(items && items.length && q.length >= minChars ? 0 : -1);
  }

  function choose(i) {
    const item = items[i];
    if (!item) return;
    close();
    onSelect(item, input);
  }

  input.addEventListener('input', update);
  input.addEventListener('focus', () => {
    // Programmatic focus (e.g. to show a field error) shouldn't cover the message with the menu.
    if (input.dataset.noAutoOpen) { delete input.dataset.noAutoOpen; return; }
    if (input.value.trim().length >= minChars || onOpenEmpty) update();
  });
  input.addEventListener('blur', () => setTimeout(close, 120));
  input.addEventListener('keydown', ev => {
    if (ev.key === 'ArrowDown') {
      ev.preventDefault();
      if (list.hidden) { update(); return; }
      if (items.length) setActive((active + 1) % items.length);
    } else if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      if (items.length) setActive((active - 1 + items.length) % items.length);
    } else if (ev.key === 'Enter') {
      if (!list.hidden && active >= 0) { ev.preventDefault(); choose(active); }
    } else if (ev.key === 'Escape') {
      if (!list.hidden) { ev.preventDefault(); ev.stopPropagation(); close(); }
    }
  });
  // mousedown (not click) so the input doesn't blur and close the list first.
  list.addEventListener('mousedown', ev => {
    const li = ev.target.closest('[data-index]');
    if (li) { ev.preventDefault(); choose(Number(li.dataset.index)); }
  });

  return { close, refresh: update };
}
