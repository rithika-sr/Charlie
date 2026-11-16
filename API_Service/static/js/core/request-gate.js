// Latest-request-wins guard. A response is applied only if no newer request started
// and nothing invalidated it (e.g. the rider edited an endpoint) while it was in flight.

export function createRequestGate() {
  let current = 0;
  let controller = null;
  return {
    /** Start a request: aborts the previous one and returns { token, signal }. */
    start() {
      controller?.abort();
      controller = new AbortController();
      return { token: ++current, signal: controller.signal };
    },
    /** Abort anything in flight and make every outstanding token stale. */
    invalidate() {
      controller?.abort();
      controller = null;
      current++;
    },
    isCurrent(token) { return token === current; },
    abort() { controller?.abort(); },
  };
}
