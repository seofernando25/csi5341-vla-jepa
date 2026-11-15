/* Story shell: top bar, chapter rail, the six chapters, footer, deep links. */

import { el, setText, icon } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";
import { CHAPTERS, parseStoryHash, storyHash, mountRail, createScrollSpy } from "./nav.js";

const CHAPTER_MODULES = [
  "./chapter-task.js",
  "./chapter-probe.js",
  "./chapter-tree.js",
  "./chapter-dreaming.js",
  "./chapter-changed.js",
  "./method.js",
];

/* Loaded one by one so a chapter still in flight leaves the shell standing. */
const MODULES = (
  await Promise.all(CHAPTER_MODULES.map((path) => import(path).catch(() => null)))
).filter((mod) => mod && typeof mod.mount === "function");

export function mountStory(root, ctx) {
  const media = window.matchMedia("(prefers-reduced-motion: reduce)");
  const story = makeStory(ctx, media);
  ctx.story = story;

  const sections = new Map();
  const handles = [];
  let active = null;

  /* --- top bar ----------------------------------------------------------- */
  const runStudy = el("span", { class: "story-run-study" });
  const runState = el("span", { class: "story-run-state" });
  const runCycle = el("span", { class: "story-run-cycle" });
  const runTime = el("span", { class: "story-run-time" });
  const separator = (after) =>
    el("span", { class: "story-run-sep", dataset: { after }, "aria-hidden": "true" }, "·");
  const sepStudy = separator("study");
  const sepState = separator("state");
  const sepCycle = separator("cycle");
  const run = el("div", { class: "story-run" }, [
    runStudy,
    sepStudy,
    runState,
    sepState,
    runCycle,
    sepCycle,
    runTime,
  ]);

  const workbench = el("a", { class: "story-switch-link", href: "#/" }, "Workbench");
  const themeBtn = el("button", { class: "story-theme", type: "button" }, icon("moon", 15));
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    ctx.dispatch(ctx.act.theme(next));
  });

  const top = el("header", { class: "story-top" }, [
    el("div", { class: "story-brand" }, [
      el("span", { class: "brand-mark", "aria-hidden": "true" }, "R"),
      el("a", { class: "story-wordmark", href: storyHash(CHAPTERS[0].id) }, "Dream-RSI"),
    ]),
    run,
    el("div", { class: "story-top-actions" }, [
      el("nav", { class: "story-switch", "aria-label": "Mode" }, [
        el("a", { class: "story-switch-link", href: storyHash(CHAPTERS[0].id), "aria-current": "page" }, "Story"),
        workbench,
      ]),
      themeBtn,
    ]),
  ]);

  /* --- rail, chapters, footer -------------------------------------------- */
  const rail = mountRail(go);
  const wait = el("p", { class: "story-wait" }, "Waiting for the run.");
  const main = el("main", { class: "story-main", id: "story-main" }, [wait]);
  for (const mod of MODULES) {
    const section = el("section", {
      class: "chapter",
      id: `chapter-${mod.id}`,
      dataset: { chapter: mod.id },
    });
    main.append(section);
    sections.set(mod.id, section);
    handles.push(mod.mount(section, ctx) || null);
  }

  const footStudy = el("span", { class: "story-footer-study" });
  const footTime = el("span", { class: "story-footer-time" });
  const footSep = el("span", { class: "story-footer-sep", "aria-hidden": "true" }, "·");
  const footer = el("footer", { class: "story-footer" }, [
    el("span", { class: "story-footer-mark" }, "Dream-RSI"),
    el("span", { class: "story-footer-line" }, [footStudy, footSep, footTime]),
  ]);

  const storyEl = el("div", { class: "story" }, [
    top,
    el("div", { class: "story-body" }, [rail.element, main, footer]),
  ]);

  /* --- top bar render ---------------------------------------------------- */
  function renderTop(state) {
    const data = state.server.data;
    const loop = data?.loop || null;
    wait.hidden = Boolean(data);
    run.hidden = !data;
    if (data) {
      setText(runStudy, data.study_dir || "");
      const label = sel.loopStatusLabel(state);
      setText(runState, loop?.stop_reason ? `${label}: ${loop.stop_reason}` : label);
      runState.dataset.status = loop?.status || "idle";
      const iteration = loop?.iteration ?? sel.currentCycle(state)?.iteration ?? null;
      const target = loop?.cycles_target;
      if (iteration == null) setText(runCycle, "");
      else if (target != null && iteration < target) setText(runCycle, `cycle ${iteration + 1} of ${target}`);
      else setText(runCycle, `cycle ${iteration + 1}`);
      const live =
        loop?.status === "running" && loop.elapsed_s != null
          ? loop.elapsed_s + Math.max(0, (Date.now() - (state.server.at || Date.now())) / 1000)
          : loop?.elapsed_s;
      setText(runTime, live == null ? "" : fmt.elapsed(live));
    }

    const segments = [runStudy, runState, runCycle, runTime];
    const separators = [sepStudy, sepState, sepCycle];
    for (let i = 0; i < separators.length; i += 1) {
      const visible = Boolean(segments[i].textContent && segments[i + 1].textContent);
      separators[i].hidden = !visible;
      separators[i].textContent = visible ? "·" : "";
    }

    const best = sel.bestOf(state);
    const cycle = best?.cycle || sel.currentCycle(state);
    const node = best?.node || (cycle?.nodes || []).find((item) => item.cell_id !== "root");
    const href =
      cycle && node
        ? `#/node/${encodeURIComponent(cycle.cycle_id)}/${encodeURIComponent(node.cell_id)}/transcript`
        : "#/";
    if (workbench.getAttribute("href") !== href) workbench.setAttribute("href", href);

    const dark = document.documentElement.dataset.theme === "dark";
    const label = dark ? "Switch to light theme" : "Switch to dark theme";
    if (themeBtn.getAttribute("aria-label") !== label) {
      themeBtn.setAttribute("aria-label", label);
      themeBtn.title = label;
      themeBtn.replaceChildren(icon(dark ? "sun" : "moon", 15));
    }

    setText(footStudy, data?.study_dir || "no study");
    setText(footTime, state.server.at ? `updated ${fmt.clock(state.server.at)}` : "");
    const footVisible = Boolean(footStudy.textContent && footTime.textContent);
    footSep.hidden = !footVisible;
    footSep.textContent = footVisible ? "·" : "";
  }

  /* --- deep links and scroll state --------------------------------------- */
  function storyContext() {
    const hash = location.hash;
    return !hash || hash === "#" || hash === "#/" || hash.startsWith("#/story");
  }

  function replaceHash(id) {
    if (root.hidden || !storyContext()) return;
    const next = storyHash(id);
    if (location.hash !== next) history.replaceState(null, "", next);
  }

  function setActive(id, write) {
    if (!id || id === active) return;
    active = id;
    rail.setActive(id);
    if (write) replaceHash(id);
  }

  let anchor = null;

  function pin(id) {
    const section = sections.get(id);
    if (!section) return;
    section.scrollIntoView({ behavior: "auto", block: "start" });
  }

  function go(chapter) {
    const id = rail.has(chapter) ? chapter : CHAPTERS[0].id;
    active = id;
    rail.setActive(id);
    replaceHash(id);
    if (!sections.has(id)) return;
    spy.lock(900);
    pin(id);
    requestAnimationFrame(() => pin(id));
    /* Chapters expand as their data lands; keep the deep link pinned briefly. */
    anchor = id;
    for (const delay of [250, 700, 1500]) setTimeout(() => anchor === id && pin(id), delay);
    setTimeout(() => {
      if (anchor === id) anchor = null;
    }, 2200);
  }

  function chapterFromHash(hash) {
    return parseStoryHash(hash)?.chapter || null;
  }

  function onHash() {
    const chapter = chapterFromHash(location.hash);
    if (chapter && chapter !== active) go(chapter);
  }
  window.addEventListener("hashchange", onHash);

  const spy = createScrollSpy({ scroll: storyEl, sections, onChange: (id) => setActive(id, true) });

  function update(state, prev) {
    renderTop(state);
    for (const handle of handles) handle?.update?.(state, prev);
    if (anchor) requestAnimationFrame(() => anchor && pin(anchor));
  }

  function destroy() {
    spy.destroy();
    window.removeEventListener("hashchange", onHash);
    for (const handle of handles) handle?.destroy?.();
    root.replaceChildren();
  }

  storyEl.addEventListener("wheel", () => (anchor = null), { passive: true });
  storyEl.addEventListener("touchstart", () => (anchor = null), { passive: true });
  root.replaceChildren(storyEl);
  update(ctx.store.getState(), null);
  const start = chapterFromHash(location.hash);
  if (start) {
    setActive(start, false);
    requestAnimationFrame(() => go(start));
  }

  return { update, destroy, go, chapterFromHash };
}

/* --- ctx.story ----------------------------------------------------------- */

function makeStory(ctx, media) {
  const dreams = new Map();
  const events = new Map();
  const artifacts = new Map();
  const contexts = new Map();

  /* One request per key; a miss is retried after a pause, a pending dream sooner. */
  function memo(map, key, load, { retryMs = 10000, pendingMs = 0 } = {}) {
    const hit = map.get(key);
    if (hit) {
      if (!hit.retryAt || Date.now() < hit.retryAt) return hit.promise;
      map.delete(key);
    }
    const entry = { promise: null, retryAt: 0 };
    entry.promise = Promise.resolve()
      .then(load)
      .catch(() => null);
    map.set(key, entry);
    entry.promise.then((value) => {
      if (map.get(key) !== entry) return;
      if (value == null) entry.retryAt = Date.now() + retryMs;
      else if (pendingMs && value.status === "pending") entry.retryAt = Date.now() + pendingMs;
    });
    return entry.promise;
  }

  const story = {
    dream(cycleId, worldId) {
      if (!cycleId || typeof ctx.api?.dream !== "function") return Promise.resolve(null);
      return memo(dreams, `${cycleId}|${worldId || ""}`, () => ctx.api.dream(cycleId, worldId), {
        pendingMs: 4000,
      });
    },
    events(source) {
      if (!source || !ctx.api) return Promise.resolve(null);
      return memo(events, source, () => ctx.api.events(source, 0));
    },
    artifact(source, name) {
      if (!source || !name || !ctx.api) return Promise.resolve(null);
      return memo(artifacts, `${source}|${name}`, () => ctx.api.artifact(source, name));
    },
    context(source) {
      if (!source || !ctx.api) return Promise.resolve(null);
      return memo(contexts, source, () => ctx.api.context(source));
    },
    open(source, view) {
      const parsed = sel.parseSource(source);
      if (!parsed) return;
      const name = view || (parsed.kind === "policy" ? "prompt" : "transcript");
      location.hash = `#/${parsed.kind}/${encodeURIComponent(parsed.cycle)}/${encodeURIComponent(
        parsed.target,
      )}/${name}`;
    },
  };
  Object.defineProperty(story, "reduced", { get: () => media.matches, enumerable: true });
  return story;
}
