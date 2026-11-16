// Zoom-dependent grouping of map markers. Pure function (no Leaflet) so it is testable.

/**
 * Greedy screen-space clustering. Points of the same `group` (e.g. a line color family)
 * within `radius` pixels of a cluster's center join it; different groups never merge,
 * so each cluster still identifies one line.
 * points: [{ id, x, y, group }] → [{ group, ids, x, y }]
 */
export function clusterPoints(points, radius = 28) {
  const clusters = [];
  const r2 = radius * radius;
  for (const p of points) {
    let best = null, bestD = Infinity;
    for (const c of clusters) {
      if (c.group !== p.group) continue;
      const d = (c.x - p.x) ** 2 + (c.y - p.y) ** 2;
      if (d <= r2 && d < bestD) { best = c; bestD = d; }
    }
    if (best) {
      best.ids.push(p.id);
      const n = best.ids.length;
      best.x += (p.x - best.x) / n;
      best.y += (p.y - best.y) / n;
    } else {
      clusters.push({ group: p.group, ids: [p.id], x: p.x, y: p.y });
    }
  }
  return clusters;
}
