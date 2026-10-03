/* Local reading aid only. Speed does not imply word/audio/cue alignment. */
(() => {
  const reader = document.getElementById("script-reader");
  const script = document.getElementById("speaking-notes");
  const play = document.getElementById("scroll-play");
  const speed = document.getElementById("scroll-speed");
  const status = document.getElementById("scroll-status");
  const saved = Number(localStorage.getItem("presenter-scroll-speed"));
  if (saved >= 4 && saved <= 48) speed.value = saved;
  let enabled = true,
    connected = false,
    ended = false,
    previous = 0,
    position = 0,
    readyAt = 0;
  function label() {
    play.textContent = ended
      ? "Restart scroll"
      : enabled
        ? "Pause scroll"
        : "Resume scroll";
    play.setAttribute("aria-pressed", String(enabled && !ended));
    const message = !connected
      ? "Waiting for slides"
      : ended
        ? "End of chapter"
        : enabled
          ? "Auto scroll · independent pace"
          : "Scroll paused";
    if (status.textContent !== message) status.textContent = message;
  }
  function pace() {
    document.getElementById("speed-value").textContent = speed.value + " px/s";
    localStorage.setItem("presenter-scroll-speed", speed.value);
  }
  speed.addEventListener("input", pace);
  pace();
  function reset() {
    position = 0;
    reader.scrollTop = 0;
    ended = false;
    readyAt = performance.now() + 1500;
    previous = 0;
    label();
  }
  play.onclick = () => {
    if (ended) {
      enabled = true;
      reset();
    } else {
      enabled = !enabled;
      position = reader.scrollTop;
      previous = 0;
      label();
    }
  };
  document.getElementById("scroll-top").onclick = reset;
  function manual() {
    enabled = false;
    ended = false;
    label();
  }
  reader.addEventListener("wheel", manual, { passive: true });
  reader.addEventListener("touchstart", manual, { passive: true });
  reader.addEventListener("pointerdown", manual);
  reader.addEventListener("keydown", (e) => {
    if (
      [
        "ArrowUp",
        "ArrowDown",
        "PageUp",
        "PageDown",
        "Home",
        "End",
        " ",
      ].includes(e.key)
    )
      manual();
  });
  function frame(now) {
    const elapsed = previous ? Math.min(100, now - previous) : 0;
    previous = now;
    if (enabled && connected && !ended && now >= readyAt && !document.hidden) {
      const line = parseFloat(getComputedStyle(script).lineHeight);
      // Space after the script lets its final line reach the same top reading line.
      const limit = Math.max(0, script.offsetHeight - line);
      position = Math.min(
        limit,
        position + (Number(speed.value) * elapsed) / 1000,
      );
      reader.scrollTop = position;
      if (position >= limit) {
        ended = true;
        label();
      }
    }
    requestAnimationFrame(frame);
  }
  window.prompterScroll = {
    chapterChanged: reset,
    setConnected(value) {
      connected = value;
      label();
    },
  };
  label();
  requestAnimationFrame(frame);
})();
