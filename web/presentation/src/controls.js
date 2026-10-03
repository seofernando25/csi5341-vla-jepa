/* Every input handler is bound exactly once. Repeated keys never queue beats. */
DATA.forEach((s, i) => {
  if (s.hidden) return;
  const b = document.createElement("button");
  b.className = "slide-link";
  b.dataset.scene = i;
  b.innerHTML = `<small>${String(sceneNumber(i)).padStart(2, "0")} · ${s.speaker}</small><br>${s.title}`;
  b.onclick = () => {
    select(i);
    document.getElementById("slides").classList.remove("open");
  };
  document.getElementById("slides").append(b);
});
document.getElementById("prev").onclick = () => advanceBeat(-1);
document.getElementById("next").onclick = () => advanceBeat(1);
document.getElementById("animate").onclick = replayBeat;
for (const id of ["menu", "closeNav"])
  document.getElementById(id).onclick = () =>
    document.getElementById("slides").classList.toggle("open");
document.getElementById("notesBtn").onclick = () => {
  const n = document.getElementById("notes");
  n.hidden = !n.hidden;
};
document.getElementById("fullscreen").onclick = () =>
  document.fullscreenElement
    ? document.exitFullscreen()
    : document.documentElement.requestFullscreen().catch(() => {});
stage.addEventListener("click", () => advanceBeat(1));
stage.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  advanceBeat(-1);
});
window.addEventListener("keydown", (e) => {
  if (e.repeat || ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName))
    return;
  const actions = {
    ArrowRight: () => advanceBeat(1),
    ArrowLeft: () => advanceBeat(-1),
    Space: () => advanceBeat(1),
    KeyN: () => document.getElementById("notesBtn").click(),
    KeyF: () => document.getElementById("fullscreen").click(),
    Home: () => resetPresenter(),
    End: () => {
      presenter.load(cueSpecs(), performance.now(), true);
      presenter.cancel();
      animation = false;
      p = 1;
      motion.chapterProgress = 1;
      motion.from = null;
      holdComparison(8);
      render();
    },
    Escape: () => {
      document.getElementById("slides").classList.remove("open");
      document.getElementById("notes").hidden = true;
    },
  };
  if (
    window.deckReadOnly &&
    ["ArrowRight", "ArrowLeft", "Space", "Home", "End"].includes(e.code)
  ) {
    e.preventDefault();
    return;
  }
  if (actions[e.code]) {
    e.preventDefault();
    actions[e.code]();
  }
});
let uiTimer;
function revealUI() {
  for (const id of ["controls", "menu", "timeline-shell"])
    document.getElementById(id).classList.add("show");
  clearTimeout(uiTimer);
  uiTimer = setTimeout(() => {
    for (const id of ["controls", "menu", "timeline-shell"])
      document.getElementById(id).classList.remove("show");
  }, 1600);
}
document.addEventListener("pointermove", (e) => {
  if (e.clientY > innerHeight - 85 || e.clientX < 45) revealUI();
});
select(0);
revealUI();
setTimeout(() => (document.getElementById("welcome").style.opacity = 0), 4000);
