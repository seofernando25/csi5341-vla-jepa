/* Shared reading pace. All relay screens may edit it; navigation stays exclusive. */
(() => {
  if (new URLSearchParams(location.search).get("view") === "backdrop") return;
  const controls = document.getElementById("controls");
  let speed = document.getElementById("scroll-speed");
  if (!speed) {
    if (!document.querySelector('meta[name="presenter-relay"]')) return;
    const label = document.createElement("label");
    label.id = "speed-label";
    label.textContent = "Global speed";
    label.htmlFor = "scroll-speed";
    speed = document.createElement("input");
    speed.type = "range";
    speed.id = "scroll-speed";
    speed.min = "4";
    speed.max = "48";
    speed.step = "2";
    speed.value = "10";
    const output = document.createElement("output");
    output.id = "speed-value";
    output.htmlFor = "scroll-speed";
    output.textContent = "10 px/s";
    label.append(speed, output);
    controls.append(label);
  }
  const label = document.getElementById("speed-label");
  speed.title = "Changes teleprompter speed on every connected computer";
  let revision = -1,
    shared = 10,
    draft = null,
    busy = false,
    timer;
  function display(value) {
    speed.value = String(value);
    document.getElementById("speed-value").textContent = value + " px/s";
    localStorage.setItem("presenter-scroll-speed", String(value));
  }
  function receive(payload) {
    const value = payload.settings?.scrollSpeed;
    if (
      !Number.isInteger(value) ||
      value < 4 ||
      value > 48 ||
      payload.settingsRevision < revision
    )
      return;
    revision = payload.settingsRevision;
    shared = value;
    if (draft === null) display(shared);
  }
  async function publish() {
    if (busy || draft === null) return;
    busy = true;
    const requested = draft;
    try {
      const response = await fetch("relay/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scrollSpeed: requested }),
        signal: AbortSignal.timeout(2500),
      });
      if (!response.ok) throw new Error("Speed not saved");
      receive(await response.json());
      label.firstChild.textContent = "Global speed";
      if (draft === requested) draft = null;
    } catch (_) {
      draft = null;
      label.firstChild.textContent = "Speed offline";
    } finally {
      busy = false;
      if (draft !== null) publish();
      else display(shared);
    }
  }
  speed.addEventListener("input", () => {
    draft = Number(speed.value);
    display(draft);
    clearTimeout(timer);
    timer = setTimeout(publish, 80);
  });
  const stream = new EventSource("relay/events");
  stream.onmessage = (event) => {
    try {
      receive(JSON.parse(event.data));
      label.firstChild.textContent = "Global speed";
    } catch (_) {}
  };
  stream.onerror = () => {
    label.firstChild.textContent = "Speed offline";
  };
})();
