<div align="center">

# Charlie

### Boston in motion — a live, AI-assisted guide to the MBTA

Live arrivals, a real-time train map, door-to-door trip planning, honest service status and an assistant you can just ask.

[![Live Demo](https://img.shields.io/badge/Live-charlie--mbta.vercel.app-000?style=for-the-badge&logo=vercel)](https://charlie-mbta.vercel.app)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask&logoColor=white)
![JavaScript](https://img.shields.io/badge/ES_modules-no_build-F7DF1E?logo=javascript&logoColor=black)
![Leaflet](https://img.shields.io/badge/Leaflet-1.9-199900?logo=leaflet&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

<img src="docs/images/dashboard.jpg" alt="Charlie overview: greeting, trip planner, your stations, subway status and alerts" width="860">

</div>

---

## Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [How it works](#how-it-works)
- [Run it locally](#run-it-locally)
- [Tests](#tests)
- [Deployment](#deployment)
- [API reference](#api-reference)
- [Known limitations](#known-limitations)
- [MLOps pipeline (offline)](#mlops-pipeline-offline)
- [Acknowledgements](#acknowledgements)

---

## Features

| View | What you get |
|---|---|
| **Overview** | Greeting and live timestamp, a one-line subway status summary, a trip planner, live arrivals for your recent or nearest stations, the top subway alerts, and quick links to every mode. |
| **Live Map** | Full-screen dark map with MBTA line paths, all 126 stations and every subway and Green Line vehicle, color-coded and pointing in its direction of travel, refreshed about every 10 s. Line and layer filters, locate, fit-network, station search, and a route-focus mode for any route (e.g. `#/map?route=66`). |
| **Station details** | Opens as a side panel on desktop and a draggable bottom sheet on phones: served lines, accessibility, arrivals grouped by line and direction with large countdowns, and active alerts. |
| **Trip Planner** | Accessible station autocomplete (or free-text addresses), swap, use-my-location and recent trips. Options are ordered by earliest arrival (re-sortable by duration, transfers or walking) and show segments, transfers, walking time, an estimated fare and alerts on the lines used. |
| **Stations** | Search all 126 subway and light-rail stations (partial, abbreviated and fuzzy matching) and filter by line. |
| **Lines & Routes** | Subway, Bus (searchable), Commuter Rail and Ferry. Each line page shows its station sequence with the next trains in both directions, alerts, the number of vehicles reporting, and a map shortcut. |
| **Service Alerts** | Prioritized by what affects riders now. Filters for all service, subway, bus, Commuter Rail and ferry, with accessibility (elevators, escalators) kept separate. Shows active periods, updates and links to the official alert. |
| **Charlie AI** | The assistant (`/api/chat`): trips (“chiswick rd to harvard”, then “what about Hynes?”), live arrivals and alerts in plain English. Trips use the Trip Planner's journey service and render as journey cards with refresh, return-trip and edit-in-planner actions. Also voice input and read-aloud. |

Built in: global search (press <kbd>/</kbd>), a live-data freshness indicator, keyboard support throughout, ARIA comboboxes, visible focus, WCAG-checked colors (line badges pick black or white text for contrast), and reduced-motion support.

---

## Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/images/live-map.jpg" alt="Live map with line paths and directional vehicle markers"><p align="center"><b>Live map</b> — line paths, stations and live vehicles</p></td>
    <td width="50%"><img src="docs/images/station-panel.jpg" alt="Park Street station panel with arrivals and alerts"><p align="center"><b>Station panel</b> — arrivals by direction, alerts, accessibility</p></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/images/trip-planner.jpg" alt="Trip planner results"><p align="center"><b>Trip planner</b> — shortest travel time first</p></td>
    <td width="50%"><img src="docs/images/line-detail.jpg" alt="Red Line station sequence with next trains"><p align="center"><b>Line detail</b> — next trains at every station</p></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/images/alerts.jpg" alt="Service alerts center"><p align="center"><b>Service alerts</b> — filters with live counts</p></td>
    <td width="50%"><img src="docs/images/ai-chat.jpg" alt="Charlie AI conversation"><p align="center"><b>Charlie AI</b> — live answers with station and line chips</p></td>
  </tr>
</table>

<p align="center"><img src="docs/images/mobile-overview.jpg" alt="Mobile overview with bottom navigation" width="300"></p>

<details>
<summary><b>Before the redesign</b></summary>

| Desktop | Tablet | Mobile |
|---|---|---|
| <img src="docs/images/before/desktop-1440.jpg" width="300"> | <img src="docs/images/before/tablet-768.jpg" width="220"> | <img src="docs/images/before/mobile-375.jpg" width="160"> |

The design audit behind the redesign is in [`docs/design-audit.md`](docs/design-audit.md).
</details>

---

## How it works

```mermaid
flowchart LR
    subgraph Browser
        UI["ES-module SPA<br/>views · panel/sheet · Leaflet map"]
        LS[("localStorage<br/>recent stations/trips,<br/>location")]
    end
    subgraph Vercel["Vercel (serverless Python)"]
        API["Flask · API_Service/app.py<br/>transit.py helpers<br/>short TTL cache + CDN headers"]
    end
    MBTA["MBTA V3 API"]
    OAI["OpenAI"]
    TR["Transitous / Google<br/>trip routing"]
    ESRI["Esri tiles"]

    UI -- "/api/*" --> API
    UI --- LS
    API --> MBTA
    API --> OAI
    API --> TR
    UI --> ESRI
```

### Live data, honestly

- **One source of truth.** Arrivals, vehicles and alerts come from the MBTA V3 API through the Flask backend, which uses a short in-process cache (5–20 s; shapes and stops for longer) and CDN cache headers. Nothing is simulated.
- **Arrival order.** Arrivals are sorted by the time that's shown: the arrival time, or the departure time for trains leaving a terminal such as Oak Grove. Departed, cancelled and skipped trips are dropped. "Boarding" appears only when the MBTA reports that status.
- **Line status counts distinct, currently active service alerts of severity 3 or higher.** An alert used to be counted once per affected stop, so the Red Line showed "52 alerts". Elevator and escalator outages, and informational notices (severity 1–2), don't change a line's status.
- **Vehicles stay where the MBTA last reported them.** Markers move only on each refresh (about every 10 s); there's no movement interpolation. If a refresh fails, the last positions stay on screen with a "Couldn't refresh" note, and polling pauses when the map isn't visible.
- **Trip options are ordered by earliest arrival.** "Shortest trip" and "Fewest transfers" badges follow transparent rules, explained in their tooltips.
- **Chat and the Trip Planner share one journey service** (`journey.py`): the same station resolution, the same providers in the same order (Google Directions when configured, then Transitous), the same alert matching. Chat replies are written from that structured result, never by the language model, and say which of these happened: places not found, a station name that needs choosing, no itineraries, or the routing service not answering.

### Frontend structure (no build step)

```
API_Service/
├── app.py                 Flask app: REST API, chat routing, provider wiring
├── journey.py             Journey service shared by chat and the trip planner
├── places.py              Station-name resolution (abbreviations, aliases, ambiguity)
├── assistant.py           Chat trip requests, follow-ups, per-user context, replies
├── arrivals.py            Chat arrival questions and replies
├── transit.py             Pure helpers: prediction timing/sorting, alert classification
├── templates/index.html   App shell (sidebar, command bar, tab bar, panel)
├── static/css/app.css     Design tokens + components + responsive rules
└── static/js/
    ├── main.js            Router, global search, freshness, location
    ├── core/              api · store · router · constants · format · search · alerts · polyline
    ├── ui/                components · combobox · panel · station-panel · trip-form · location · toast · icons
    ├── map/live-map.js    Leaflet controller (lazy-loaded): shapes, stations, vehicle diffing, polling
    └── views/             overview · map · trip · stations · lines · alerts · assistant
```

---

## Run it locally

You need Python 3.11+, a free [MBTA V3 API key](https://api-v3.mbta.com/), and an [OpenAI API key](https://platform.openai.com/api-keys) for the assistant.

```bash
git clone https://github.com/Gnanasudharsan/Charlie.git
cd Charlie/API_Service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `API_Service/.env` (it's git-ignored):

```env
MBTA_API_KEY=your_mbta_key
OPENAI_API_KEY=your_openai_key
# Optional: the trip planner falls back to Transitous without it
GOOGLE_MAPS_API_KEY=your_google_maps_key
```

```bash
python app.py        # → http://localhost:5001
```

| Variable | Required | Used for |
|---|---|---|
| `MBTA_API_KEY` | Recommended | Live predictions, vehicles and alerts. Without it, the MBTA allows only a small number of requests per minute, and the UI shows "Couldn't refresh" when it's rate-limited. |
| `OPENAI_API_KEY` | Optional | Helps Charlie AI understand unusual phrasing (structured intent only). Trips, arrivals and alerts are parsed and answered without it. |
| `GOOGLE_MAPS_API_KEY` | Optional | Google transit directions. Without it, the trip planner uses Transitous (free, no key). |

---

## Tests

```bash
cd API_Service
pip install -r requirements.txt pytest
python -m pytest                          # backend tests: MBTA, routing and OpenAI mocked, no network or keys
node --test tests/js/*.test.mjs           # frontend logic tests (Node 20+)
```

The same commands run in CI on every pull request ([`app-tests.yml`](.github/workflows/app-tests.yml)) with no credentials.

- **Backend tests** cover prediction timing and sorting (departure-only, null, tie, UTC-offset, departed and cancelled cases), alert classification and priority, the honest line status, and that MBTA errors surface as errors rather than empty lists. They also check the station, line, dashboard, alerts, stations-list, chat and trip-planner endpoints, and that every module is served versioned (never stale).
- **Chat and journey tests** cover the reported “chiswick rd to harvard/hynes” failures, ordinary phrasings and abbreviations, follow-ups (“what about Hynes?”, “how do I get back?”), ambiguous and unknown places, router outages versus empty results, Google-to-Transitous fallback, depart-at and arrive-by times (including a missed deadline), accessibility requests, stale and per-user context, and that chat and the Trip Planner return the same itineraries for the same request, with or without a requested time.
- **Frontend tests** cover station search (every one of the 126 stations is found by name, plus partial and fuzzy queries), countdown states, alert filters and counts, polyline decoding, the router, trip ordering and badges, station mentions in AI replies, chat journey cards and their actions, arrival grouping, and badge text contrast.
- **UI flows** were checked by hand in a real browser at 375, 390, 768, 1024 and 1440 px. These covered keyboard navigation, comboboxes, map selection and filters, the bottom sheet, and the error states.

---

## Deployment

Production runs on **[Vercel](https://vercel.com)** as a serverless Python function: **https://charlie-mbta.vercel.app**

- **Every push to `main` deploys.** The Vercel project is connected to this repo, with its Root Directory set to `API_Service`. Static assets are cache-busted with the commit SHA.
- **Secrets are Vercel environment variables.** `MBTA_API_KEY`, `OPENAI_API_KEY` and optionally `GOOGLE_MAPS_API_KEY`. The live app reads keys only from the environment.
- **Zero config.** Vercel detects the Flask `app` in `app.py`, and `.vercelignore` keeps the tests out of the function bundle.

```bash
vercel link                                   # from the repo root
vercel env add MBTA_API_KEY --environment production
vercel env add OPENAI_API_KEY --environment production
vercel deploy --prod
```

<details>
<summary><b>Legacy: Google Cloud Run</b></summary>

The app was first deployed to Cloud Run with Docker (`API_Service/Dockerfile`) and GitHub Actions (`.github/workflows/deploy.yml`, `deploy-cloudrun.yml`). Those workflows now run only on manual dispatch. They need billing enabled on the GCP project and the `GCP_SA_KEY`, `MBTA_API_KEY` and `OPENAI_API_KEY` secrets.
</details>

---

## API reference

All endpoints return JSON. Base URL: `https://charlie-mbta.vercel.app`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health and version |
| `GET` | `/api/dashboard` | Per-line status (distinct alerts, active service alerts, accessibility alerts, vehicles reporting) and severe alerts |
| `GET` | `/api/alerts` | All alerts, prioritized, with `active_now`, `active_period`, `modes`, `category` (`service`/`accessibility`), `url` |
| `GET` | `/api/station/<stop_id>` | Station details, accessibility, lines, time-sorted predictions (`time`, `event`, `status`, direction, platform) and active alerts |
| `GET` | `/api/line/<route_id>` | Line info, ordered `stations` with next trains per direction, sorted `predictions_by_stop`, vehicles and alerts |
| `GET` | `/api/vehicles?route=<ids>&limit=` | Live vehicle positions with bearing, headsign, current stop and report time |
| `GET` | `/api/shapes?route=<ids>` | Encoded route polylines for the map |
| `GET` | `/api/stations` | All 126 stations with coordinates and lines |
| `GET` | `/api/routes` | Routes grouped by mode |
| `GET` | `/api/predictions?stop=<id>` | Upcoming predictions at a stop (or `?route=<id>`) |
| `GET` | `/api/nearest-station?lat=&lon=` | Nearest station to a coordinate |
| `POST` | `/api/set-location` | Resolve a coordinate to its nearest station |
| `POST` | `/api/trip-planner` | Directions. Body: `{ origin, destination, origin_id?, dest_id?, origin_coords?, dest_coords? }`; `code` is `ok`, `ambiguous`, `unresolved`, `same_place`, `no_itineraries` or `routing_unavailable` (HTTP 502) |
| `POST` | `/api/chat` | Assistant. Body: `{ message, history?, location?, context?, action? }`; returns `{ reply, history, context }` plus `journey`, `arrivals` or `alerts` when the answer is about one |

Changes to the API only add fields; existing field names are unchanged.

---

## Known limitations

- **Trip legs aren't drawn on the map.** The routing response's leg geometry isn't passed through yet. Fares are estimates.
- **Bus, Commuter Rail and ferry vehicles appear only in route-focus mode.** The main map shows subway and light rail; open a route's page and choose "Show on live map" to see its vehicles.
- **Commuter Rail and ferry service is mostly scheduled.** Live predictions appear only when the MBTA publishes them.
- **Times chat can't read, and step-free routing, aren't applied.** “By 5 pm”, “tomorrow at 9am” or “in 30 minutes” are sent to the router, like the Trip Planner's Depart at / Arrive by; a phrase such as “next Monday”, or “step-free route”, is answered for leaving now with a note saying so. Bus and Commuter Rail stop questions that need the language model are still phrased by it from live predictions.
- **Dark theme only.** No Playwright suite yet; UI flows were verified manually in the browser.
- **The ML model isn't used by the live app.** The delay-prediction model in [`Model_Development/`](Model_Development/readme.md) is part of the offline pipeline below.

---

## MLOps pipeline (offline)

The repo also contains the course MLOps pipeline that predicts train-delay probability from MBTA event data. **It runs separately from the live web app.**

| Stage | Where | What |
|---|---|---|
| Data pipeline | [`Data_Pipeline/`](Data_Pipeline/readme.md) | Airflow DAGs to collect, process and validate MBTA data (TFDV and Great Expectations), with DVC versioning and Fairlearn bias checks |
| Model development | [`Model_Development/`](Model_Development/readme.md) | Logistic Regression and LightGBM, tuning with SMOTE, MLflow tracking, SHAP and LIME, fairness, Evidently drift. Tuned Logistic Regression was selected (ROC-AUC **0.81**, per `reports/model_comparison.json`) |
| Monitoring & retraining | [`monitoring/`](monitoring/README.md), [`retraining/`](retraining/) | Prometheus-style metrics. Retraining triggers on PSI drift and promotes a model only if its AUC improves |
| CI | [`.github/workflows/charlie_mbta_ci.yml`](.github/workflows/charlie_mbta_ci.yml) | Train, tune, select, register, run bias, explainability and drift checks, and upload the reports |

---

## Acknowledgements

- Transit data from the [MBTA V3 API](https://www.mbta.com/developers/v3-api). Charlie is not affiliated with the MBTA.
- Trip routing by [Transitous](https://transitous.org), built on [MOTIS](https://github.com/motis-project/motis) and open GTFS data.
- Map tiles © [Esri](https://www.esri.com) — Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors.
- Built for **IE 7374 — MLOps** at Northeastern University (Group 22).

Released under the [MIT License](LICENSE).
