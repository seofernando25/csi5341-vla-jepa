/* Pure deterministic state machine. No DOM, drawing, audio or media dependencies. */
class CueClock {
  constructor() {
    this.manual = true;
    this.cancel();
    this.cursor = 0;
    this.seconds = 0;
    this.specs = [];
  }
  load(specs, now = 0, atEnd = false) {
    this.cancel();
    this.specs = specs;
    this.cursor = atEnd ? specs.length - 1 : 0;
    this.seconds = atEnd ? specs.at(-1).end : specs[0].start;
    if (!atEnd) this.begin(0, 1, now);
    else {
      this.run = {
        from: this.seconds,
        to: this.seconds,
        began: now,
        duration: 600,
        direction: 0,
      };
      this.running = true;
      this.progress = 0;
    }
  }
  get spec() {
    return this.specs[this.cursor];
  }
  begin(cursor, direction, now) {
    this.cursor = cursor;
    const spec = this.spec;
    this.run = {
      from: direction > 0 ? spec.start : spec.end,
      to: direction > 0 ? spec.end : spec.start,
      began: now,
      duration: spec.duration,
      direction,
    };
    this.seconds = this.run.from;
    this.running = true;
    this.progress = 0;
  }
  next(now) {
    if (this.running) return "busy";
    if (this.cursor + 1 === this.specs.length) return "chapter";
    this.begin(this.cursor + 1, 1, now);
    return "beat";
  }
  previous(now) {
    if (this.running) return "busy";
    if (this.cursor === 0) return "chapter";
    this.begin(this.cursor, -1, now);
    return "beat";
  }
  replay(now) {
    if (this.running) return "busy";
    this.begin(this.cursor, 1, now);
    return "beat";
  }
  tick(now) {
    if (!this.running) return false;
    const run = this.run,
      t = Math.max(0, Math.min(1, (now - run.began) / run.duration));
    this.progress = t;
    this.seconds = run.from + (run.to - run.from) * t;
    if (t === 1) {
      this.running = false;
      if (run.direction < 0) {
        this.cursor--;
        this.seconds = this.spec.end;
      }
      this.run = null;
    }
    return true;
  }
  cancel() {
    this.running = false;
    this.run = null;
    this.progress = 1;
  }
}
