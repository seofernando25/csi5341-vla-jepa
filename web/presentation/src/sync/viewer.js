/* Relay-served plain route follows the controller. Static hosting stays standalone. */
(() => {
  if (
    !document.querySelector('meta[name="presenter-relay"]') ||
    new URLSearchParams(location.search).get("relay") === "control"
  )
    return;
  window.deckReadOnly = true;
  for (const button of document.querySelectorAll(
    "#prev,#next,#animate,.slide-link,[data-chapter]",
  )) {
    button.disabled = true;
    if (["prev", "next", "animate"].includes(button.id)) button.hidden = true;
  }
  document.getElementById("menu").hidden = true;
  document.getElementById("welcome").textContent =
    "Synced viewer · navigation follows the slide controller";
  document.getElementById("caption").textContent =
    "Read-only synchronized slide viewer";
  const status = document.createElement("span");
  status.style.cssText = "font-size:12px;color:#62748a;padding:0 8px";
  status.setAttribute("role", "status");
  status.id = "viewer-status";
  document.getElementById("controls").append(status);
  let lastSnapshot = "",
    lastMessage = 0;
  function apply(state, ageMs = 0) {
    if (index !== state.scene) select(state.scene);
    presenter.cancel();
    presenter.cursor = state.cue;
    const spec = cueSpecs()[state.cue];
    presenter.seconds = Number.isFinite(state.seconds)
      ? state.seconds
      : spec.end;
    presenter.progress = state.progress ?? 1;
    if (state.running && state.remaining > 0) {
      const target =
        state.direction < 0
          ? spec.start
          : state.direction === 0
            ? presenter.seconds
            : spec.end;
      const fraction = Math.min(1, Math.max(0, ageMs / state.remaining));
      presenter.seconds += (target - presenter.seconds) * fraction;
      presenter.progress += (1 - presenter.progress) * fraction;
      const remaining = state.remaining * (1 - fraction);
      if (remaining > 0) {
        presenter.run = {
          from: presenter.seconds,
          to: target,
          began: performance.now(),
          duration: remaining,
          direction: state.direction,
          progressFrom: presenter.progress,
        };
        presenter.running = true;
      } else if (state.direction < 0 && presenter.cursor > 0) {
        presenter.cursor--;
        presenter.seconds = presenter.spec.end;
      }
    }
    syncPresenterMedia();
    syncPresenterFrame();
  }
  const stream = new EventSource("relay/events");
  stream.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      lastMessage = Date.now();
      status.textContent = payload.controllerOnline
        ? "Synced · read-only viewer"
        : "Controller offline · holding last state";
      if (
        !payload.state ||
        !SCENE_CUES[payload.state.scene]?.[payload.state.cue]
      )
        return;
      const serialized = JSON.stringify(payload.state);
      if (serialized !== lastSnapshot) {
        apply(payload.state, payload.ageMs);
        lastSnapshot = serialized;
      }
    } catch (_) {
      status.textContent = "Invalid relay state";
    }
  };
  stream.onerror = () => {
    lastSnapshot = "";
    status.textContent = "Relay disconnected · reconnecting";
  };
  setInterval(() => {
    if (lastMessage && Date.now() - lastMessage > 6500)
      status.textContent = "Relay disconnected · reconnecting";
  }, 1000);
})();
