/* SSE transport with bounded reconnect. Emits parsed frames; never touches the DOM. */

const MAX_RETRIES = 6;

export function openStream({ source, after = 0, onEvent, onStatus }) {
  let es = null;
  let closed = false;
  let retries = 0;
  let cursor = after;
  let timer = null;

  const status = (value, detail) => onStatus && onStatus(value, detail);

  function connect() {
    if (closed) return;
    status(retries ? "reconnecting" : "connecting");
    const url = `/api/stream?source=${encodeURIComponent(source)}&after=${cursor}`;
    try {
      es = new EventSource(url);
    } catch (error) {
      status("error", String(error));
      return;
    }
    es.addEventListener("hello", (frame) => {
      retries = 0;
      const payload = parse(frame);
      if (payload && typeof payload.cursor === "number") cursor = payload.cursor;
      status("open", payload);
      onEvent({ type: "hello", payload });
    });
    es.addEventListener("events", (frame) => {
      const payload = parse(frame);
      if (!payload) return;
      if (typeof payload.next === "number") cursor = payload.next;
      onEvent({ type: "events", events: payload.events || [], next: cursor });
    });
    es.addEventListener("state", (frame) => {
      const payload = parse(frame);
      if (payload) onEvent({ type: "state", payload });
    });
    es.addEventListener("reset", (frame) => {
      const payload = parse(frame);
      cursor = 0;
      onEvent({ type: "reset", reason: payload?.reason || "reset" });
    });
    es.addEventListener("done", (frame) => {
      const payload = parse(frame);
      if (payload && typeof payload.next === "number") cursor = payload.next;
      onEvent({ type: "done", reason: payload?.reason || "settled", next: cursor });
      close();
    });
    es.addEventListener("error", () => {
      if (closed) return;
      if (es) es.close();
      es = null;
      retries += 1;
      if (retries > MAX_RETRIES) {
        status("error", "stream unavailable");
        onEvent({ type: "error", error: "stream unavailable" });
        return;
      }
      status("reconnecting");
      timer = setTimeout(connect, Math.min(5000, 300 * 2 ** retries));
    });
  }

  function close() {
    closed = true;
    if (timer) clearTimeout(timer);
    if (es) es.close();
    es = null;
  }

  connect();
  return { close, get cursor() { return cursor; } };
}

function parse(frame) {
  try {
    return JSON.parse(frame.data);
  } catch {
    return null;
  }
}
