// Browser geolocation → nearest station (via /api/set-location). Location stays in this browser.

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { toast } from './toast.js';

let pending = null;

export function requestLocation({ quiet = false } = {}) {
  if (pending) return pending;
  if (!navigator.geolocation) {
    toast('Location isn’t available in this browser.', { tone: 'error' });
    return Promise.resolve(null);
  }
  document.body.classList.add('is-locating');
  pending = new Promise(resolve => {
    navigator.geolocation.getCurrentPosition(async pos => {
      const { latitude: lat, longitude: lon } = pos.coords;
      const { data } = await api.setLocation(lat, lon);
      const location = (data && data.location) || { lat, lon, station: null, lines: [] };
      store.set('location', location);
      if (!quiet) {
        toast(location.station ? `You’re near ${location.station} (${location.distance} m)` : 'No MBTA station within 2 km of you.',
          { tone: location.station ? 'success' : 'info' });
      }
      resolve(location);
    }, err => {
      const msg = err.code === err.PERMISSION_DENIED ? 'Location permission was denied.' :
        err.code === err.TIMEOUT ? 'Finding your location timed out.' : 'Couldn’t find your location.';
      toast(msg, { tone: 'error' });
      resolve(null);
    }, { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 });
  }).finally(() => { pending = null; document.body.classList.remove('is-locating'); });
  return pending;
}
