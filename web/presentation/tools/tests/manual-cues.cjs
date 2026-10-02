const fs = require("node:fs"),
  vm = require("node:vm"),
  assert = require("node:assert/strict"),
  path = require("node:path");
const root = path.resolve(__dirname, "../.."),
  context = vm.createContext({});
for (const file of ["timing/cues.js", "timing/clock.js"])
  vm.runInContext(
    fs.readFileSync(path.join(root, "src", file), "utf8"),
    context,
  );
const specs = vm.runInContext("SCENE_CUES", context),
  Clock = vm.runInContext("CueClock", context);
let tested = 0;
for (const [scene, beats] of Object.entries(specs)) {
  const clock = new Clock();
  let now = 0;
  clock.load(beats, now);
  for (let j = 0; j < beats.length; j++) {
    if (j) assert.equal(clock.next(now), "beat");
    assert.equal(clock.cursor, j);
    assert.equal(
      clock.next(now + 1),
      "busy",
      "Rapid next must not queue or skip",
    );
    assert.equal(clock.previous(now + 1), "busy");
    assert.equal(clock.replay(now + 1), "busy");
    const spec = beats[j];
    assert(spec.end > spec.start);
    assert(spec.duration >= 500 && spec.duration <= 5900);
    clock.tick(now + spec.duration / 2);
    assert(clock.seconds >= spec.start && clock.seconds <= spec.end);
    now += spec.duration;
    clock.tick(now);
    assert.equal(clock.seconds, spec.end);
    assert.equal(clock.running, false);
    const held = clock.seconds;
    clock.tick(now + 30000);
    assert.equal(clock.seconds, held, "No automatic changes while held");
    assert.equal(clock.replay(now), "beat");
    now += spec.duration;
    clock.tick(now);
    assert.equal(
      clock.seconds,
      held,
      "Replay must restore the same final state",
    );
    tested++;
  }
  assert.equal(clock.next(now), "chapter");
  for (let j = beats.length - 1; j > 0; j--) {
    assert.equal(clock.previous(now), "beat");
    now += beats[j].duration;
    clock.tick(now);
    assert.equal(clock.cursor, j - 1);
    assert.equal(clock.seconds, beats[j - 1].end);
  }
  assert.equal(clock.previous(now), "chapter");
}
assert.equal(Object.keys(specs).length, 18);
const changes = { 2: [6.1], 3: [9, 36, 55], 9: [10, 20, 34], 11: [24] };
for (const [scene, times] of Object.entries(changes)) {
  assert(specs[scene][0].end < times[0]);
  for (const spec of specs[scene])
    assert(times.filter((t) => t >= spec.start && t <= spec.end).length <= 1);
}
const media = fs.readFileSync(path.join(root, "src/media.js"), "utf8");
const mediaContext = vm.createContext({document:{createElement:()=>({setAttribute(){},style:{}}),body:{append(){}}},Image:function(){},sceneSeconds:()=>12});
vm.runInContext(media,mediaContext);
assert.equal(vm.runInContext("comparisonTime()",mediaContext),12);
assert.equal(vm.runInContext("comparisonExportTime=3.2;comparisonTime()",mediaContext),3.2);
console.log(
  `18 chapters; ${tested} beats tested forward, backward and replay; rapid input and held states checked.`,
);

// Returning to a chapter holds its last beat while the chapter bridge animates.
for (const beats of Object.values(specs)) {
  const c = new Clock();
  c.load(beats, 100, true);
  assert.equal(c.cursor, beats.length - 1);
  assert.equal(c.seconds, beats.at(-1).end);
  assert.equal(c.next(101), "busy");
  c.tick(700);
  assert.equal(c.running, false);
  assert.equal(c.next(700), "chapter");
}
