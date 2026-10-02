/* Authoring-only exhaustive transition sampler. Uses the real CueClock and scene renderer. */
const reviewButton = document.createElement("button");
reviewButton.id = "review-transitions";
reviewButton.textContent = "Audit every transition";
document.getElementById("notes").append(reviewButton);
reviewButton.onclick = async () => {
  reviewButton.disabled = true;
  audio.pause();
  presenter.manual = true;
  await document.fonts.ready;
  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 720;
  const c = canvas.getContext("2d");
  sketch.pixelDensity(1);
  sketch.resizeCanvas(1280, 720);
  const post = async (name, body) => {
    const r = await fetch("/export/" + name, { method: "POST", body });
    if (!r.ok) throw Error("Audit export " + r.status);
  };
  const seek = async (time) => {
    comparisonExportTime = time;
    await Promise.all(
      [humanClip, robotClip].map(async (v) => {
        v.pause();
        const t = Math.min(time, Math.max(0, v.duration - 0.04));
        if (!Number.isFinite(t) || Math.abs(v.currentTime - t) < 0.001) return;
        await new Promise((resolve) => {
          v.addEventListener("seeked", resolve, { once: true });
          v.currentTime = t;
        });
      }),
    );
  };
  const take = async (name) => {
    animation = false;
    sketch.redraw();
    c.drawImage(sketch.canvas, 0, 0, 1280, 720);
    await post(
      name,
      await new Promise((resolve) => canvas.toBlob(resolve, "image/png")),
    );
  };
  const rows = [];
  let now = 0;
  try {
    for (const i of visibleScenes) {
      index = i;
      motion.from = null;
      motion.dataFrom = null;
      motion.oldHeader = "";
      presenter.load(SCENE_CUES[i], now);
      presenter.cancel();
      for (let j = 0; j < SCENE_CUES[i].length; j++)
        for (const direction of ["forward", "back", "replay"]) {
          if (direction === "back" && j === 0) continue;
          presenter.begin(j, direction === "back" ? -1 : 1, now);
          const spec = SCENE_CUES[i][j];
          for (let f = 0; f <= 4; f++) {
            presenter.tick(now + (spec.duration * f) / 4);
            syncPresenterFrame();
            animation = false;
            if (i === 0 || i === 2) await seek(presenter.seconds);
            await take(
              `review-${String(sceneNumber(i)).padStart(2, "0")}-${j}-${direction}-${f}.png`,
            );
          }
          rows.push({
            chapter: sceneNumber(i),
            beat: j + 1,
            label: spec.label,
            direction,
            durationMs: spec.duration,
          });
          now += spec.duration;
        }
      reviewButton.textContent = `Audited chapter ${sceneNumber(i)} / 18`;
    }
    // Actual chapter changes in each direction; preserve headers, poses and benchmark values.
    for (const direction of [1, -1]) {
      const scenes =
        direction > 0 ? visibleScenes : [...visibleScenes].reverse();
      for (let j = 1; j < scenes.length; j++) {
        index = scenes[j - 1];
        presenter.load(SCENE_CUES[index], now, true);
        presenter.tick(now + 600);
        p = presenter.seconds / motionLength();
        motion.from = null;
        motion.oldHeader = "";
        animation = false;
        sketch.redraw();
        select(scenes[j], direction < 0);
        const spec = presenter.run;
        now = spec.began;
        for (let f = 0; f <= 4; f++) {
          presenter.tick(now + (spec.duration * f) / 4);
          syncPresenterFrame();
          animation = false;
          if (index === 0 || index === 2) await seek(presenter.seconds);
          await take(
            `review-${String(sceneNumber(index)).padStart(2, "0")}-chapter-${direction > 0 ? "forward" : "back"}-${f}.png`,
          );
        }
        rows.push({
          chapter: sceneNumber(index),
          beat: "chapter",
          direction: direction > 0 ? "forward" : "back",
          durationMs: spec.duration,
        });
        now += spec.duration;
      }
    }
    await post(
      "review-report.json",
      new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" }),
    );
    reviewButton.textContent = "All transitions audited";
  } catch (e) {
    reviewButton.textContent = "Audit failed: " + e.message;
    console.error(e);
  } finally {
    comparisonExportTime = null;
    presenter.manual = true;
    select(3);
    sketch.resizeCanvas(innerWidth, innerHeight);
    reviewButton.disabled = false;
  }
};
