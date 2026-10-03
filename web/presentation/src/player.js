/* Scene selection and note hydration only. Input is owned by controls.js. */
const visibleScenes = ACTIVE_CHAPTERS;
function sceneNumber(i) {
  return chapterNumber(i);
}
function render() {
  if (window.updateTimeline) window.updateTimeline();
  if (window.publishPresenterState) window.publishPresenterState();
}
function select(n, atEnd = false) {
  beginTransform(n);
  audio.pause();
  animation = false;
  index = n;
  p = 0;
  if (index === 0) {
    motion.header = "";
    motion.oldHeader = "";
  }
  for (const v of [humanClip, robotClip]) v.pause();
  audio.src = `assets/narration-${String(index + 1).padStart(2, "0")}.mp3?v=story-20261002-v4`;
  document
    .querySelectorAll(".slide-link")
    .forEach((b) =>
      b.classList.toggle("active", Number(b.dataset.scene) === index),
    );
  document.getElementById("noteTitle").textContent =
    "Chapter " + sceneNumber(index) + " · " + DATA[index].speaker;
  document.getElementById("script").textContent = DATA[index].notes;
  document.getElementById("delivery").textContent = (
    DATA[index].deliveryCues || []
  )
    .map(
      (c) =>
        Math.floor(c.time / 60) +
        ":" +
        String(Math.floor(c.time % 60)).padStart(2, "0") +
        " — " +
        c.instruction,
    )
    .join("\n");
  document.getElementById("source").textContent = DATA[index].source;
  document.getElementById("caption").textContent =
    "Right / Space: next beat · Left: previous beat · Chapters: jump";
  setLiveMedia(index, true);
  resetPresenter(atEnd);
  render();
}
function moveScene(d) {
  const pos = visibleScenes.indexOf(index),
    next = pos + d;
  if (next >= 0 && next < visibleScenes.length)
    select(visibleScenes[next], d < 0);
}
window.deck = {
  select,
  render,
  setProgress(v) {
    presenter.cancel();
    p = clamp(v);
    render();
  },
  get index() {
    return index;
  },
  get progress() {
    return p;
  },
  data: DATA,
};
