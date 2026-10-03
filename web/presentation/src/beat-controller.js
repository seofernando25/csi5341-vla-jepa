/* Browser adapter for the pure clock. Chapter selection and controls stay elsewhere. */
const presenter = new CueClock();
function cueSpecs(i = index) {
  return SCENE_CUES[i];
}
function cueTimes(i = index) {
  return cueSpecs(i).map((s) => s.end);
}
function syncPresenterFrame() {
  motion.chapterProgress =
    presenter.cursor === 0 || presenter.run?.direction === 0
      ? presenter.progress
      : 1;
  p = clamp(presenter.seconds / motionLength());
  animation = presenter.running;
  render();
}
function syncPresenterMedia() {
  if (
    presenter.manual &&
    motion.exportTime === null &&
    comparisonExportTime === null
  ) {
    setLiveMedia(index);
    return;
  }
  if (index === 0 || index === 2) {
    const run = presenter.run;
    syncMedia(
      presenter.seconds,
      presenter.running && run?.direction > 0,
      run ? Math.abs(run.to - run.from) / (run.duration / 1000) : 1,
    );
  }
}
function resetPresenter(atEnd = false) {
  presenter.manual = true;
  presenter.load(cueSpecs(), performance.now(), atEnd);
  syncPresenterMedia();
  syncPresenterFrame();
}
function advanceBeat(direction = 1) {
  if (window.deckReadOnly) return;
  audio.pause();
  const result =
    direction > 0
      ? presenter.next(performance.now())
      : presenter.previous(performance.now());
  if (result === "busy" && direction > 0) {
    presenter.accelerate(performance.now());
  }
  if (result === "chapter") moveScene(direction);
  else if (result === "beat") {
    syncPresenterMedia();
    syncPresenterFrame();
  }
}
function replayBeat() {
  if (window.deckReadOnly) return;
  if (presenter.replay(performance.now()) === "beat") {
    syncPresenterMedia();
    syncPresenterFrame();
  }
}
function tickPresenter(now) {
  if (animation && presenter.manual && presenter.tick(now)) {
    syncPresenterFrame();
    if (!presenter.running) syncPresenterMedia();
  }
}
