# Charlie redesign — design audit

_Audit of the production site (charlie-mbta.vercel.app) and `API_Service/` on 2026-11-16, before the redesign._

## Current experience

- **Stack:** a Flask app (`API_Service/app.py`) renders one Jinja template (`templates/index.html`). It is driven by a single 1,600-line global script (`static/script.js`) and a 2,100-line stylesheet. There's no build step, framework or router. Tabs are toggled with `display`. Leaflet 1.9 comes from unpkg, with Esri Dark Gray tiles.
- **Navigation:** 8 horizontal tabs (Overview, Chat, Map, Subway, Bus, Rail, Ferry, Alerts). On mobile they shrink to unlabeled icons and scroll sideways off-screen.
- **Overview:** the trip planner, then "System Status" (vehicle and alert counts per line), then alerts, then station lookup at the bottom of the page.
- **Map:** a fixed-height card showing stations and live trains (full re-render every 10 s). Popups use small buttons, and there are no line filters or detail panel.
- **Lines:** 4 separate tabs, each with its own copy of the same detail renderer.
- **Station details:** a modal.
- **Chat:** a working AI assistant with voice input.

## Main problems

1. **Misleading numbers.** "System Status" counts an alert once per *affected stop*, so the Red Line shows "52 alerts" when there are only a handful. Elevator closures count as "severe" line alerts and turn lines red.
2. **The Lines view's arrivals aren't sorted.** Per-stop predictions come from the MBTA's `arrival_time` sort, so departure-only trips are out of order. This is the same bug fixed earlier in the station view.
3. **Weak hierarchy.** Station lookup, the most common task, is at the bottom of the page. Nothing answers "how is my line right now?" at a glance.
4. **Mobile isn't designed.** The scrolling icon tabs have no labels, the map is a small card, and the modal covers the whole screen.
5. **The map is an afterthought.** It has no line paths, no filters, no freshness indicator, and markers are rebuilt on every refresh.
6. **Accessibility gaps.** Inline `onclick` handlers on `div`s, no keyboard support in the dropdowns, color-only meaning, a disabled zoom (`user-scalable=no`), no focus styles, and no live regions.
7. **Inconsistent components.** Emoji used as icons, four different card styles, and the alert filter's active state never matches the "Rail" button.
8. **Performance.** The dashboard makes 8 MBTA vehicle requests one after another, and nothing is cached, including the static station list.

## Proposed information architecture

| Primary view | Contents |
|---|---|
| **Overview** | Greeting, live freshness, line status strip, compact trip form, your/nearby stations with live arrivals, priority alerts, quick access |
| **Live Map** | Full-bleed map: line paths, stations, live vehicles, layer/line filters, recenter, locate, detail panel/sheet |
| **Trip Planner** | Origin/destination comboboxes, swap, locate, recent trips, ranked option cards with alert indicators and step details |
| **Stations** | Searchable list of all 126 stations with line filters and recents; station detail panel |
| **Lines & Routes** | Subway / Bus / Commuter Rail / Ferry segments; line detail with station sequence, next trains, alerts, map focus |
| **Service Alerts** | Prioritized, filterable alerts (mode filters plus a separate accessibility filter), active periods, source links |
| **Charlie AI** | Conversational assistant with suggestions, voice input, read-aloud, station and line chips |

On desktop: a left sidebar plus a top command bar with global search. On mobile: a compact header plus a bottom tab bar (Home, Map, Plan, Lines, Charlie). Alerts and Stations are reachable from the header. The details panel is a right-hand drawer on desktop and a draggable bottom sheet on mobile, and it appears only when something is selected.

## Component strategy

No framework migration. The Flask template stays. The frontend becomes native ES modules (no bundler):

- `js/core/`: `api` (fetch with abort and timeouts), `store` (tiny pub/sub), `router` (hash routes), `constants` (line colors and names), `format` (time, countdown, escaping), `search` (ranking), `alerts` (classification and priority).
- `js/ui/`: `components` (LineBadge, ArrivalRow, AlertCard, TripOptionCard, EmptyState, ErrorState, Skeleton), `combobox` (accessible autocomplete), `panel` (drawer/sheet), `toast`.
- `js/views/`: one module per primary view.
- `js/map/`: lazy-loaded Leaflet controller (layers, vehicle diffing, polling).
- `css/`: design tokens plus components.

## Risks to existing functionality

- **Station lookup, sorted arrivals, Esri tiles, vehicle markers, trip-planner fallback and per-user chat state.** All are covered by new pytest and Node tests and checked in the browser.
- **The chat API contract** (`message`, `history`, `location`) stays unchanged.
- **API changes are additive only.** Existing fields keep their names, and new fields are added alongside.
- **Shipping.** It goes to production on a push to `main`, so it's only pushed after a local end-to-end check.

## Intended visual improvements

- **Palette and type:** a calm dark palette (canvas `#090D14` and surfaces `#111827` to `#243044`), Inter with tabular numerals, and line colors used only for transit meaning, always paired with a text label.
- **Countdowns:** large countdowns, with honest states ("Boarding" only when the MBTA status says so, "Departs" for departure-only trips).
- **Map:** line paths drawn from MBTA shapes, crisp directional vehicle chevrons that update in place, and a selected state.
- **Motion:** subtle 150–200 ms motion, respecting `prefers-reduced-motion`. No idle pulsing.

## Out of scope / future

- **Trip geometry on the map:** the routing API's leg geometry isn't exposed yet.
- **Light theme and Playwright CI:** Playwright needs a browser download in CI.
- **Bus and commuter-rail vehicles on the main map:** available in route-focus mode only.
