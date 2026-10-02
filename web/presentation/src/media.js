const robotClip = document.createElement("video");
robotClip.id = "robot-dataset-clip";
robotClip.src = "assets/droid-demonstration.mp4";
robotClip.muted = true;
robotClip.loop = true;
robotClip.playsInline = true;
robotClip.preload = "auto";
robotClip.setAttribute("aria-hidden", "true");
robotClip.style.display = "none";
document.body.append(robotClip);
const robotPoster = new Image();
robotPoster.src = "assets/droid-demonstration-poster.jpg";
const humanClip = document.createElement("video");
humanClip.id = "human-dataset-clip";
humanClip.src = "assets/human-demonstration.mp4";
humanClip.muted = true;
humanClip.loop = true;
humanClip.playsInline = true;
humanClip.preload = "auto";
humanClip.style.display = "none";
humanClip.setAttribute("aria-hidden", "true");
document.body.append(humanClip);
let comparisonExportTime = null;
function comparisonTime() {
  return comparisonExportTime ?? sceneSeconds();
}
function holdComparison(t) {
  syncMedia(t);
}

function syncMedia(time, play = false, rate = 1) {
  for (const v of [humanClip, robotClip]) {
    if (Number.isFinite(v.duration))
      v.currentTime = Math.min(time, Math.max(0, v.duration - 0.04));
    v.playbackRate = Math.max(0.25, Math.min(4, rate));
    if (play) v.play().catch(() => {});
    else v.pause();
  }
}

// Live footage has its own clock: cue advances never seek or speed up a video.
function setLiveMedia(chapter, reset = false) {
  for (const v of [humanClip, robotClip]) {
    const active = chapter === 2 || (chapter === 0 && v === robotClip);
    v.loop = true;
    v.playbackRate = 1;
    if (reset) v.currentTime = 0;
    if (active) v.play().catch(() => {});
    else v.pause();
  }
}
