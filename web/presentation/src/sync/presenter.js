/* Read-only teleprompter: scripts map to chapters, never words or animation time. */
(() => {
  const el = (id) => document.getElementById(id);
  const text = (id, value) => {
    if (el(id).textContent !== value) el(id).textContent = value;
  };
  const slideSource = el("slide-source");
  const togglePreview = el("toggle-preview");
  const preview = el("slide-preview");
  const previewCanvas = preview.querySelector("canvas");
  const previewContext = previewCanvas.getContext("2d", { alpha: false });
  const slideFrame = slideSource.querySelector("iframe");
  function mirrorSlide() {
    if (!preview.hidden && !document.hidden) {
      const source = slideFrame.contentDocument?.querySelector("canvas");
      if (source?.width && source?.height)
        previewContext.drawImage(
          source,
          0,
          0,
          previewCanvas.width,
          previewCanvas.height,
        );
    }
    requestAnimationFrame(mirrorSlide);
  }
  requestAnimationFrame(mirrorSlide);
  preview.hidden = localStorage.getItem("presenter-hide-slide") === "true";
  function previewLabel() {
    togglePreview.textContent = preview.hidden
      ? "Show preview"
      : "Hide preview";
    togglePreview.setAttribute("aria-pressed", String(!preview.hidden));
  }
  togglePreview.onclick = () => {
    preview.hidden = !preview.hidden;
    localStorage.setItem("presenter-hide-slide", String(preview.hidden));
    previewLabel();
  };
  previewLabel();
  const size = el("text-size");
  const stored = Number(localStorage.getItem("presenter-text-size"));
  if (stored >= 24 && stored <= 64) size.value = stored;
  function resize() {
    document.documentElement.style.setProperty(
      "--notes-size",
      size.value + "px",
    );
    el("size-value").textContent = size.value + " px";
    localStorage.setItem("presenter-text-size", size.value);
  }
  size.addEventListener("input", resize);
  resize();
  el("full").onclick = () =>
    document.fullscreenElement
      ? document.exitFullscreen()
      : document.documentElement.requestFullscreen().catch(() => {});
  let scene = null,
    received = false,
    lastMessage = 0;
  function renderScript(notes) {
    el("speaking-notes").replaceChildren();
    for (const paragraph of notes.split(/\n\s*\n/)) {
      const p = document.createElement("p");
      ReadingAssist.render(p, paragraph);
      el("speaking-notes").append(p);
    }
  }
  ReadingAssist.bind(el("reading-aid"), () => {
    const reader = el("script-reader");
    const position = reader.scrollTop;
    if (scene !== null) renderScript(DATA[scene].notes);
    reader.scrollTop = position;
  });
  function connection(text, online = false) {
    if (el("connection").textContent !== text)
      el("connection").textContent = text;
    el("connection").dataset.state = online ? "connected" : "waiting";
    window.prompterScroll?.setConnected(online);
  }
  function update(payload) {
    lastMessage = Date.now();
    connection(
      payload.controllerOnline
        ? "Connected · following slide computer"
        : "Relay connected · slide computer offline",
      payload.controllerOnline,
    );
    const state = payload.state;
    if (
      !state ||
      !DATA[state.scene] ||
      DATA[state.scene].hidden ||
      !SCENE_CUES[state.scene]?.[state.cue]
    )
      return;
    received = true;
    const data = DATA[state.scene],
      cues = SCENE_CUES[state.scene];
    const number = chapterNumber(state.scene);
    text(
      "chapter",
      `Chapter ${number} / ${ACTIVE_CHAPTERS.length} · slide ${number}`,
    );
    text("speaker", data.speaker);
    text("title", data.title);
    text(
      "cue",
      `Cue ${state.cue + 1} / ${cues.length} · ${cues[state.cue].label}`,
    );
    text("motion", state.running ? "Animation playing" : "Holding");
    el("last-state").textContent = payload.controllerOnline
      ? ""
      : "Showing the last received chapter and cue. They may be out of date.";
    if (scene === state.scene) return; // A new cue must never jump the reader's script/scroll.
    scene = state.scene;
    renderScript(data.notes);
    el("delivery-notes").textContent = (data.deliveryCues || [])
      .map((c) => c.instruction)
      .join("\n\n");
    el("delivery").hidden = !data.deliveryCues?.length;
    el("delivery").open = false;
    el("script-reader").scrollTop = 0;
    window.prompterScroll?.chapterChanged();
  }
  const stream = new EventSource("relay/events");
  stream.onmessage = (event) => {
    try {
      update(JSON.parse(event.data));
    } catch (_) {
      connection("Invalid relay message");
    }
  };
  function disconnected() {
    connection("Disconnected · reconnecting automatically");
    el("last-state").textContent = received
      ? "Showing the last received chapter and cue. They may be out of date."
      : "No local relay found. Run tools/relay.py on the slide computer; use the printed Notes URL. Manual slide notes remain available with N.";
  }
  stream.onerror = disconnected;
  // Detect a silent broken Wi-Fi link even before the OS notices the dropped socket.
  setInterval(() => {
    if (lastMessage && Date.now() - lastMessage > 6500) disconnected();
  }, 1000);
})();
