/* Story chapters, the story hash route, the rail and the scroll spy. */

import { el } from "../dom.js";

export const CHAPTERS = [
  {
    id: "task",
    number: "01",
    title: "The task",
    dek: "A LIBERO instruction, ten trials, and a policy that must act on what it sees.",
  },
  {
    id: "probe",
    number: "02",
    title: "The probe",
    dek: "Each candidate is a smaller policy, scored on a fixed probe budget.",
  },
  {
    id: "tree",
    number: "03",
    title: "The tree",
    dek: "Branches, attempts, and the order the search actually took.",
  },
  {
    id: "dreaming",
    number: "04",
    title: "Dreaming",
    dek: "When the cycle ends, the run replays its own steps and picks a recipe.",
  },
  {
    id: "changed",
    number: "05",
    title: "What changed",
    dek: "What the cycle changed, in the policy and in the numbers.",
  },
  {
    id: "method",
    number: "06",
    title: "Method",
    dek: "How these numbers are produced, and what they do not say.",
  },
];

const BY_ID = new Map(CHAPTERS.map((chapter) => [chapter.id, chapter]));

export function parseStoryHash(hash = location.hash) {
  const text = String(hash || "").replace(/^#\/?/, "");
  const [root, chapter] = text.split("/");
  if (root !== "story" || !chapter) return null;
  let id;
  try {
    id = decodeURIComponent(chapter);
  } catch {
    return null;
  }
  return BY_ID.has(id) ? { chapter: id } : null;
}

export function storyHash(chapter = CHAPTERS[0].id) {
  return `#/story/${encodeURIComponent(chapter)}`;
}

/* Rail: numbers and titles, active chapter marked. */
export function mountRail(onGo) {
  const links = new Map();
  const list = el("ol", { class: "story-rail-list" });
  for (const chapter of CHAPTERS) {
    const link = el("a", { class: "story-rail-link", href: storyHash(chapter.id) }, [
      el("span", { class: "story-rail-number" }, chapter.number),
      el("span", { class: "story-rail-title" }, chapter.title),
    ]);
    link.addEventListener("click", (event) => {
      event.preventDefault();
      onGo(chapter.id);
    });
    links.set(chapter.id, link);
    list.append(el("li", { class: "story-rail-item" }, link));
  }
  const element = el("nav", { class: "story-rail", "aria-label": "Chapters" }, [
    el("p", { class: "story-rail-head" }, "Chapters"),
    list,
  ]);
  return {
    element,
    has: (id) => links.has(id),
    setActive(id) {
      for (const [key, link] of links) {
        if (key === id) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      }
    },
  };
}

/* Scroll spy: the last chapter whose top has crossed the bar. */
export function createScrollSpy({ scroll, sections, onChange, line = 72 }) {
  let frame = 0;
  let lockUntil = 0;

  function compute() {
    const top = scroll.getBoundingClientRect().top + line;
    let current = sections.keys().next().value || null;
    for (const [id, section] of sections) {
      if (section.getBoundingClientRect().top <= top) current = id;
    }
    return current;
  }

  function schedule() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (performance.now() < lockUntil) return;
      onChange(compute());
    });
  }

  function unlock() {
    lockUntil = 0;
    schedule();
  }

  function onKey(event) {
    if (["ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", " "].includes(event.key)) unlock();
  }

  const observer = new IntersectionObserver(schedule, {
    root: scroll,
    rootMargin: `-${line}px 0px -55% 0px`,
    threshold: [0, 0.01, 0.25, 0.5, 1],
  });
  for (const section of sections.values()) observer.observe(section);
  scroll.addEventListener("scroll", schedule, { passive: true });
  scroll.addEventListener("wheel", unlock, { passive: true });
  scroll.addEventListener("touchstart", unlock, { passive: true });
  scroll.addEventListener("keydown", onKey);
  window.addEventListener("resize", schedule);

  return {
    schedule,
    lock(ms) {
      lockUntil = performance.now() + ms;
    },
    destroy() {
      observer.disconnect();
      if (frame) cancelAnimationFrame(frame);
      scroll.removeEventListener("scroll", schedule);
      scroll.removeEventListener("wheel", unlock);
      scroll.removeEventListener("touchstart", unlock);
      scroll.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", schedule);
    },
  };
}
