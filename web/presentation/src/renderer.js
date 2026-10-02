/* Display refresh and canvas framing. Playback decisions belong to CueClock. */
new p5((q) => {
  sketch = q;
  q.setup = () => {
    q.pixelDensity(Math.min(devicePixelRatio || 1, 2));
    let cn = q.createCanvas(innerWidth, innerHeight);
    cn.parent(stage);
    cn.elt.setAttribute("aria-label", "Animated VLA-JEPA presentation");
    canvasCtx = q.drawingContext;
    q.noLoop();
    requestAnimationFrame(function repaint() {
      q.redraw();
      requestAnimationFrame(repaint);
    });
  };
  q.windowResized = () => q.resizeCanvas(innerWidth, innerHeight);
  q.draw = () => {
    const now = performance.now();
    tickPresenter(now);
    if (lastPaint) {
      frameTimes.push(now - lastPaint);
      if (frameTimes.length > 180) frameTimes.shift();
    }
    lastPaint = now;
    framesSeen++;
    const c = canvasCtx,
      scale = Math.min(q.width / 960, q.height / 540),
      dx = (q.width - 960 * scale) / 2,
      dy = (q.height - 540 * scale) / 2;
    c.save();
    c.fillStyle = "white";
    c.fillRect(0, 0, q.width, q.height);
    c.translate(dx, dy);
    c.scale(scale, scale);
    drawScene(c);
    c.restore();
    q.canvas.dataset.progress = p.toFixed(4);
    q.canvas.dataset.scene = String(sceneNumber(index));
    q.canvas.dataset.cue = String(presenter.cursor + 1);
    q.canvas.dataset.running = String(presenter.running);
  };
}, stage);
