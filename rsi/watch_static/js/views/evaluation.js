/* Evaluation: score hero, protocol facts, fail class, error text and raw score.json. */

import { el, clear, icon, copyButton } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

export function mount(root, ctx) {
  const view = el("div", { class: "view evaluation-view" });
  root.append(view);
  let sig = "";
  let rawOpen = false;

  function render(state) {
    const tab = sel.activeTab(state);
    const source = tab ? tab.id : null;
    const node = tab && tab.kind === "node" ? sel.nodeForSource(state, source) : null;
    const gate = state.server.data?.protocol?.smoke_gate ?? null;
    const scoreArt = source ? sel.artifact(state, source, "score") : null;
    const errorArt = source ? sel.artifact(state, source, "error") : null;
    const next = JSON.stringify([
      source,
      state.server.status,
      state.server.error,
      sel.loopLive(state),
      node && [
        node.status,
        node.fail_class,
        node.score,
        node.size_reduction,
        node.successes,
        node.episodes,
        node.resident_mb,
        node.commit,
        node.agent_seconds,
        node.eval_seconds,
        node.agent_cost_usd,
        node.agent_turns,
        node.agent_timed_out,
        node.round,
        node.branch,
        node.attempt,
        node.error,
        node.promoted,
      ],
      gate,
      scoreArt && [scoreArt.status, scoreArt.at],
      errorArt && [errorArt.status, errorArt.at],
    ]);
    if (next === sig) return;
    sig = next;
    clear(view);

    if (!state.server.data) {
      const failed = state.server.status === "error";
      view.append(
        el("p", { class: "empty" }, [
          el("strong", {}, failed ? "State unavailable" : "Loading state…"),
          failed ? state.server.error || "The server did not answer." : null,
        ]),
      );
      return;
    }
    if (!tab) {
      view.append(el("p", { class: "empty" }, [el("strong", {}, "No node open"), "Select a node in the explorer."]));
      return;
    }
    if (tab.kind !== "node") {
      view.append(
        el("p", { class: "empty" }, [
          el("strong", {}, "Not a node"),
          "Evaluation belongs to a discovery node.",
        ]),
      );
      return;
    }
    if (!node) {
      view.append(el("p", { class: "empty" }, [el("strong", {}, "Node not in this cycle"), source]));
      return;
    }
    renderNode(state, node, gate, scoreArt, errorArt);
  }

  function renderNode(state, node, gate, scoreArt, errorArt) {
    const failClass = node.fail_class;
    const badge = failClass
      ? el("span", { class: "pill", dataset: { tone: failTone(failClass) } }, fmt.failLabel(failClass))
      : el(
          "span",
          { class: "pill", dataset: { tone: phaseTone(state, node) } },
          fmt.nodeStatusLabel(sel.nodePhase(state, node)),
        );

    const smokeTone = node.successes == null || gate == null ? null : node.successes >= gate ? "ok" : "err";

    view.append(
      el("section", { class: "eval-hero" }, [
        el("div", { class: "eval-score" }, [
          el("span", { class: "eval-score-value num" }, fmt.pct(node.score)),
          el("span", { class: "eval-score-label" }, "score"),
        ]),
        badge,
        el("div", { class: "eval-hero-stats" }, [
          heroStat(
            "smoke",
            `${node.successes ?? "—"} / ${node.episodes ?? "—"}`,
            gate == null ? null : `gate ${gate}`,
            smokeTone,
          ),
          heroStat("size", fmt.pct(node.size_reduction)),
          heroStat("resident", fmt.mb(node.resident_mb)),
        ]),
      ]),
    );

    if (failClass && failClass !== "ok") {
      view.append(
        el("div", { class: "eval-fail", dataset: { tone: failTone(failClass) } }, [
          el("span", { class: "eval-fail-label" }, fmt.failLabel(failClass)),
        ]),
      );
    }

    const rawError =
      errorArt && errorArt.status === "ready" && errorArt.text && errorArt.text.trim()
        ? errorArt.text
        : node.error;
    if (rawError) {
      view.append(
        el("section", { class: "eval-error" }, [
          el("div", { class: "eval-error-head" }, [
            icon("alert", 12),
            el("span", {}, "Error"),
            failClass ? el("span", { class: "eval-error-class" }, fmt.failLabel(failClass)) : null,
          ]),
          el("pre", { class: "eval-error-text mono", tabIndex: 0 }, rawError),
        ]),
      );
    }

    view.append(el("h3", { class: "eval-section-head" }, "Facts"));
    const facts = el("div", { class: "eval-facts" });
    facts.append(
      factCard("agent time", fmt.dur(node.agent_seconds)),
      factCard("eval time", fmt.dur(node.eval_seconds)),
      factCard("agent cost", fmt.usd(node.agent_cost_usd)),
      factCard("turns", node.agent_turns == null ? "—" : String(node.agent_turns)),
      factCard("timed out", node.agent_timed_out ? "yes" : "no", { tone: node.agent_timed_out ? "err" : null }),
      factCard("round", node.round == null ? "—" : String(node.round)),
      factCard("branch / attempt", `${node.branch ?? "—"} / ${node.attempt ?? "—"}`),
      factCard("commit", node.commit ? fmt.shortId(node.commit, 7) : "—", {
        mono: true,
        title: node.commit || null,
      }),
    );
    if (node.promoted) {
      facts.append(
        factCard("promoted (dev)", `${node.promoted.successes ?? "—"} / ${node.promoted.episodes ?? "—"}`),
      );
    }
    view.append(facts);
    view.append(rawScore(scoreArt));
  }

  function rawScore(entry) {
    const details = el("details", { class: "eval-raw", open: rawOpen });
    const stateLabel = artifactLabel(entry);
    details.append(
      el("summary", {}, [
        icon("chevronRight", 11),
        el("span", { class: "eval-raw-name" }, "score.json"),
        el("span", { class: "eval-raw-state" }, stateLabel),
      ]),
    );
    details.addEventListener("toggle", () => {
      rawOpen = details.open;
    });
    if (entry && entry.status === "ready" && entry.text != null) {
      const text = pretty(entry.text);
      details.append(
        el("div", { class: "eval-raw-bar" }, [copyButton(() => text, "Copy")]),
        el("pre", { class: "eval-json mono", tabIndex: 0 }, text),
      );
    } else {
      const message = entry && entry.status === "loading" ? "loading…" : "not recorded";
      const note = el("p", { class: "eval-raw-empty", title: entry?.error || null }, message);
      details.append(note);
    }
    return details;
  }

  render(ctx.store.getState());
  return { update: render, destroy() {} };
}

function heroStat(label, value, note, tone) {
  return el("div", { class: "eval-hero-stat", dataset: tone ? { tone } : null }, [
    el("span", { class: "eval-hero-label" }, label),
    el("span", { class: "eval-hero-value num" }, value),
    note ? el("span", { class: "eval-hero-note" }, note) : null,
  ]);
}

function factCard(label, value, opts = {}) {
  return el("div", { class: "eval-fact", dataset: opts.tone ? { tone: opts.tone } : null }, [
    el("span", { class: "eval-fact-label" }, label),
    el(
      "span",
      {
        class: `eval-fact-value${opts.mono ? " mono" : ""}`,
        title: opts.title || null,
      },
      value,
    ),
  ]);
}

function pretty(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return String(text);
  }
}

function artifactLabel(entry) {
  if (!entry) return "not loaded";
  if (entry.status === "ready") return "recorded";
  if (entry.status === "loading") return "loading";
  return "not recorded";
}

function failTone(failClass) {
  if (failClass === "ok") return "ok";
  if (failClass === "gate_near_miss" || failClass === "gate_failed") return "warn";
  return "err";
}

function phaseTone(state, node) {
  const phase = sel.nodePhase(state, node);
  if (phase === "agent" || phase === "evaluating") return "live";
  if (phase === "stopped") return "warn";
  return "idle";
}
