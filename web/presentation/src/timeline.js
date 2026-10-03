/* Chapter rail is a view of presenter state; it does not bind playback handlers. */
const mainScenes = visibleScenes,
  totalDuration = mainScenes.reduce((v, i) => v + DATA[i].budget, 0);
const fmt = (t) =>
  Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
for (const i of mainScenes) {
  const b = document.createElement("button");
  b.style.flex = DATA[i].budget;
  b.dataset.chapter = i;
  b.title = DATA[i].title;
  b.setAttribute("aria-label", "Chapter " + sceneNumber(i) + " · " + b.title);
  b.onclick = () => select(i);
  document.getElementById("chapter-timeline").append(b);
}
window.updateTimeline = () => {
  document.getElementById("chapter-label").textContent =
    "Chapter " + sceneNumber(index) + " / " + visibleScenes.length;
  document.getElementById("clock-label").textContent =
    presenter.spec?.label || "";
  for (const b of document.querySelectorAll("[data-chapter]")) {
    const i = Number(b.dataset.chapter);
    b.classList.toggle("current", i === index);
    b.classList.toggle("visited", i < index);
    b.style.setProperty(
      "--position",
      ((presenter.cursor + 1) / cueSpecs().length) * 100 + "%",
    );
  }
};
