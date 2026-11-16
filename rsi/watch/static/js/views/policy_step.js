/* Policy step views: prompt, diff, replay, verdict.
   The diff and syntax engines belong to other modules; both are optional here so a missing
   engine degrades to plain text instead of breaking the view. */
import { el, clear, icon, copyButton } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

const highlightReady = import("../highlight.js").catch(() => null);
const diffReady = import("../diff.js").catch(() => null);
const VERDICT_TONE = {
  accepted: "ok",
  not_better: "warn",
  timeout: "warn",
  timeout_salvaged: "warn",
  rejected: "err",
  running: "live",
};

/* --- shared chrome ------------------------------------------------------- */

function verdictTone(status) {
  return VERDICT_TONE[status] || "";
}

function deltaTone(value) {
  if (value == null || Number.isNaN(Number(value))) return "";
  return value > 0 ? "ok" : value < 0 ? "err" : "";
}

function pill(status) {
  return el("span", { class: "pill", dataset: { tone: verdictTone(status) } }, fmt.verdictLabel(status));
}

function empty(title, detail) {
  return el("p", { class: "empty" }, [el("strong", {}, title), detail || null]);
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function isObj(value) {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function viewHead(source, step, status = null) {
  const parsed = source ? sel.parseSource(source) : null;
  const label = status || step?.status;
  return el("header", { class: "pol-head" }, [
    el("span", { class: "pol-title" }, step ? `Policy step ${step.step}` : `Policy step ${parsed?.target || ""}`),
    label ? pill(label) : null,
    step?.mean_v != null ? el("span", { class: "pol-headmeta num", title: "Mean V" }, `V ${fmt.num(step.mean_v)}`) : null,
    el("span", { class: "pol-source mono subtle truncate", title: source || "" }, source || ""),
  ]);
}

function sizeText(section) {
  const parts = [];
  if (section.chars) parts.push(`${fmt.chars(section.chars)} chars`);
  if (section.tokens) parts.push(fmt.tokens(section.tokens));
  return parts.join(" · ");
}

function paint(node, code, lang = "python") {
  node.textContent = code;
  highlightReady.then((mod) => {
    if (!mod || typeof mod.highlightInto !== "function" || !node.isConnected) return;
    try {
      mod.highlightInto(node, code, lang);
      if (!node.firstChild) node.textContent = code;
    } catch {
      node.textContent = code;
    }
  });
}

function codeBlock(code, lang = "python") {
  const pre = el("pre", { class: "pol-code" });
  paint(pre, code, lang);
  return pre;
}

/* --- prompt -------------------------------------------------------------- */

function sectionHead(title, size, text) {
  const summary = el("summary", { class: "pol-sum" }, [
    icon("chevronRight", 11),
    el("span", { class: "pol-sectitle truncate" }, title),
    size ? el("span", { class: "pol-size subtle num" }, size) : null,
  ]);
  if (text) {
    summary.append(
      el("span", { class: "pol-copy", on: { click: (event) => event.stopPropagation() } }, copyButton(() => text, "Copy")),
    );
  }
  return summary;
}

function fieldsNode(fields) {
  const dl = el("dl", { class: "pol-fields" });
  for (const [key, value] of Object.entries(fields)) {
    if (value == null || value === "") continue;
    const text = typeof value === "boolean" ? (value ? "yes" : "no") : String(value);
    dl.append(el("dt", { class: "mono" }, key), el("dd", { class: "wrap-any", title: text.length > 40 ? text : null }, text));
  }
  return dl;
}

function itemNode(item) {
  const text = item.text || "";
  return el("details", { class: "pol-item" }, [
    sectionHead(item.name || "item", `${fmt.chars(item.chars ?? text.length)} chars`, text),
    el("div", { class: "pol-secbody" }, codeBlock(text, /\.py$/.test(item.name || "") ? "python" : "text")),
  ]);
}

function sectionNode(section) {
  const title = section.title || section.id || "Section";
  if (section.recorded === false) {
    return el("section", { class: "pol-section pol-missing" }, [
      el("div", { class: "pol-sum" }, [el("span", { class: "pol-sectitle" }, title), el("span", { class: "pol-flag" }, "not recorded")]),
      el("p", { class: "pol-note" }, section.note || "Not recorded."),
    ]);
  }
  const body = el("div", { class: "pol-secbody" });
  if (section.items?.length) {
    for (const item of section.items) body.append(itemNode(item));
  } else if (section.fields && Object.keys(section.fields).length) {
    body.append(fieldsNode(section.fields));
  } else if (section.text) {
    const python = section.id === "seed" || section.id === "candidate" || /\.py$/.test(section.path || "");
    body.append(codeBlock(section.text, python ? "python" : "text"));
  } else {
    body.append(el("p", { class: "pol-note" }, "Empty."));
  }
  const details = el("details", { class: "pol-section", open: true }, [
    sectionHead(title, sizeText(section), section.text || ""),
    body,
  ]);
  if (section.path) details.title = section.path;
  return details;
}

function fallbackSections(payload) {
  const out = (payload.files || []).map((file) => ({
    id: file.name,
    title: file.name,
    recorded: true,
    text: file.text,
    chars: (file.text || "").length,
    tokens: Math.ceil((file.text || "").length / 4),
    path: file.path,
    items: [],
    fields: {},
  }));
  const fields = Object.fromEntries(
    Object.entries(payload.command || {}).filter(([, value]) => value != null && !Array.isArray(value)),
  );
  if (Object.keys(fields).length) {
    out.push({ id: "command", title: "Agent configuration", recorded: true, text: null, chars: 0, tokens: 0, path: null, items: [], fields });
  }
  return out;
}

export function mountPrompt(root, ctx) {
  const wrap = el("div", { class: "pol pol-prompt" });
  root.append(wrap);
  let sig = "";
  function render(state) {
    const source = sel.activeSource(state);
    const entry = source ? sel.contextFor(state, source) : null;
    const step = source ? sel.stepForSource(state, source) : null;
    const next = `${source}|${entry?.status}|${entry?.at}|${step?.status}|${step?.mean_v}`;
    if (next === sig) return;
    sig = next;
    clear(wrap);
    if (!source) return wrap.append(empty("Select a policy step."));
    wrap.append(viewHead(source, step));
    if (!entry) return wrap.append(empty("No prompt recorded.", "The prompt is read from the step's logs."));
    if (entry.status === "loading") return wrap.append(empty("Loading prompt…"));
    if (entry.status === "error") return wrap.append(empty("Prompt not recorded.", entry.error));
    const payload = entry.payload || {};
    const sections = payload.sections?.length ? payload.sections : fallbackSections(payload);
    if (!sections.length) return wrap.append(empty("No prompt material recorded."));
    const list = el("div", { class: "pol-sections" });
    for (const section of sections) list.append(sectionNode(section));
    wrap.append(list);
  }
  render(ctx.store.getState());
  return { update: render, destroy() {} };
}

/* --- diff ---------------------------------------------------------------- */

function flattenDiff(file) {
  const lines = [];
  let adds = 0;
  let dels = 0;
  for (const hunk of file?.hunks || []) {
    lines.push({ type: "hunk", text: hunk.header || "" });
    for (const line of hunk.lines || []) {
      const type = line.kind === "add" ? "add" : line.kind === "del" ? "del" : line.kind === "meta" ? "meta" : "ctx";
      if (type === "add") adds += 1;
      else if (type === "del") dels += 1;
      lines.push({ type, text: String(line.text ?? ""), oldNo: line.oldNo ?? null, newNo: line.newNo ?? null });
    }
  }
  return { lines, adds, dels };
}

function diffRows(lines, seedText, candText, hl) {
  let oldRows = null;
  let newRows = null;
  try {
    if (hl && typeof hl.highlightLines === "function") {
      oldRows = hl.highlightLines(seedText, "python");
      newRows = hl.highlightLines(candText, "python");
    }
  } catch {
    oldRows = null;
    newRows = null;
  }
  const rows = el("div", { class: "pol-diffrows" });
  for (const line of lines) {
    if (line.type === "hunk" || line.type === "meta") {
      rows.append(el("div", { class: "pol-dline", dataset: { type: line.type } }, el("span", { class: "pol-hunk mono" }, line.text || "")));
      continue;
    }
    const code = el("code", { class: "pol-dcode" });
    const tokens = line.type === "add" ? newRows?.[line.newNo - 1] : oldRows?.[line.oldNo - 1];
    if (tokens) code.append(tokens);
    else code.textContent = line.text;
    rows.append(
      el("div", { class: "pol-dline", dataset: { type: line.type } }, [
        el("span", { class: "pol-ln num" }, line.oldNo == null ? "" : String(line.oldNo)),
        el("span", { class: "pol-ln num" }, line.newNo == null ? "" : String(line.newNo)),
        el("span", { class: "pol-sign" }, line.type === "add" ? "+" : line.type === "del" ? "-" : ""),
        code,
      ]),
    );
  }
  return rows;
}

export function mountDiff(root, ctx) {
  const wrap = el("div", { class: "pol pol-diff" });
  root.append(wrap);
  let sig = "";
  function render(state) {
    const source = sel.activeSource(state);
    const seed = source ? sel.artifact(state, source, "seed") : null;
    const cand = source ? sel.artifact(state, source, "policy") : null;
    const step = source ? sel.stepForSource(state, source) : null;
    const next = `${source}|${seed?.status}|${seed?.at}|${cand?.status}|${cand?.at}|${step?.status}`;
    if (next === sig) return;
    sig = next;
    clear(wrap);
    if (!source) return wrap.append(empty("Select a policy step."));
    wrap.append(viewHead(source, step));
    if (cand?.status === "loading" || seed?.status === "loading") return wrap.append(empty("Loading diff…"));
    if (!cand || cand.status !== "ready" || !cand.text) {
      return wrap.append(empty("No candidate code recorded for this step.", cand?.error || "candidate.py is missing."));
    }
    const seedText = seed?.status === "ready" ? seed.text || "" : "";
    const body = el("div", { class: "pol-diffbody" });
    wrap.append(body);
    Promise.all([diffReady, highlightReady]).then(([mod, hl]) => {
      if (!body.isConnected) return;
      let file = null;
      try {
        file = mod && typeof mod.diffLines === "function" ? mod.diffLines(seedText, cand.text, { path: "candidate.py" }) : null;
      } catch {
        file = null;
      }
      const diff = flattenDiff(file);
      if (!diff.lines.length) {
        body.append(el("p", { class: "pol-note" }, "Diff unavailable; showing the candidate."));
        body.append(codeBlock(cand.text));
        return;
      }
      body.append(
        el("div", { class: "pol-diffhead" }, [
          el("span", { class: "mono" }, "seed"),
          el("span", { class: "pol-arrow" }, "→"),
          el("span", { class: "mono" }, "candidate.py"),
          el("span", { class: "pol-diffstat num" }, [
            el("span", { class: "pol-add" }, `+${diff.adds}`),
            el("span", { class: "pol-del" }, `-${diff.dels}`),
          ]),
        ]),
      );
      if (!seedText) body.append(el("p", { class: "pol-note" }, "Incumbent seed not recorded; the candidate is shown as additions."));
      body.append(diffRows(diff.lines, seedText, cand.text, hl));
    });
  }
  render(ctx.store.getState());
  return { update: render, destroy() {} };
}

/* --- replay -------------------------------------------------------------- */

function cellText(value) {
  if (value == null) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
  if (Array.isArray(value)) return `${value.length} rows`;
  if (isObj(value)) return JSON.stringify(value).slice(0, 40);
  const text = String(value);
  return text.length > 48 ? `${text.slice(0, 47)}…` : text;
}

function tablesFrom(data) {
  if (Array.isArray(data)) return data.length && data.every(isObj) ? [{ title: null, rows: data }] : [];
  if (!isObj(data)) return [];
  const tables = [];
  for (const [key, value] of Object.entries(data)) {
    if (Array.isArray(value) && value.length && value.every(isObj)) {
      tables.push({ title: key, rows: value });
    } else if (isObj(value) && Object.keys(value).length && Object.values(value).every(isObj)) {
      tables.push({
        title: key,
        rows: Object.entries(value).map(([id, row]) => {
          const flat = { id };
          for (const [name, cell] of Object.entries(row)) if (!isObj(cell)) flat[name] = Array.isArray(cell) ? `${cell.length} rows` : cell;
          return flat;
        }),
      });
    }
  }
  return tables;
}

function tableView(title, rows) {
  const shown = rows.slice(0, 50);
  const cols = [];
  for (const row of shown) for (const key of Object.keys(row)) if (!cols.includes(key)) cols.push(key);
  const head = cols.slice(0, 8);
  const table = el("table", { class: "pol-table" }, [
    el("thead", {}, el("tr", {}, head.map((col) => el("th", { class: "mono" }, col)))),
    el(
      "tbody",
      {},
      shown.map((row) =>
        el(
          "tr",
          {},
          head.map((col) => {
            const cell = el("td", { class: "num" }, cellText(row[col]));
            if (typeof row[col] === "string" && row[col].length > 24) cell.title = row[col];
            return cell;
          }),
        ),
      ),
    ),
  ]);
  return el("section", { class: "pol-tablewrap" }, [
    title ? el("div", { class: "pol-tabletitle" }, [el("span", {}, title), el("span", { class: "subtle num" }, `${rows.length} rows`)]) : null,
    el("div", { class: "pol-tablescroll" }, table),
    rows.length > shown.length ? el("div", { class: "pol-more subtle" }, `${rows.length - shown.length} more rows`) : null,
  ]);
}

function metaRow(entries) {
  return el(
    "div",
    { class: "pol-meta" },
    entries.map(([key, value]) =>
      el("span", { class: "pol-chip", title: key }, [
        el("i", {}, key),
        el("b", { class: "num" }, typeof value === "string" && value.length > 16 ? fmt.shortId(value) : cellText(value)),
      ]),
    ),
  );
}

function rawBlock(text, label) {
  return el("details", { class: "pol-raw" }, [
    sectionHead(label, `${fmt.chars(text.length)} chars`, text),
    el("pre", { class: "pol-json" }, text),
  ]);
}

function cards(items) {
  return el(
    "div",
    { class: "pol-cards" },
    items.map(([label, value, tone]) =>
      el("div", { class: "pol-card", dataset: { tone } }, [
        el("span", { class: "pol-cardlabel" }, label),
        el("span", { class: "pol-cardvalue num" }, value),
      ]),
    ),
  );
}

export function mountReplay(root, ctx) {
  const wrap = el("div", { class: "pol pol-replay" });
  root.append(wrap);
  let sig = "";
  function render(state) {
    const source = sel.activeSource(state);
    const parsed = source ? sel.parseSource(source) : null;
    const dev = (parsed ? sel.cycleById(state, parsed.cycle) : null)?.policy_dev || null;
    const step = source ? sel.stepForSource(state, source) : null;
    const curve = source ? sel.artifact(state, source, "curve") : null;
    const next = `${source}|${curve?.status}|${curve?.at}|${step?.status}|${step?.mean_v}|${step?.delta_vs_best}|${dev?.heldout_delta}`;
    if (next === sig) return;
    sig = next;
    clear(wrap);
    if (!source) return wrap.append(empty("Select a policy step."));
    wrap.append(viewHead(source, step));
    const parsedCurve = curve?.status === "ready" ? parseJson(curve.text) : null;
    const curveObj = isObj(parsedCurve) ? parsedCurve : null;
    wrap.append(
      cards([
        ["mean V", fmt.num(step?.mean_v ?? curveObj?.mean_v), ""],
        ["delta vs best", fmt.signed(step?.delta_vs_best), deltaTone(step?.delta_vs_best)],
        ["held-out delta", fmt.signed(dev?.heldout_delta), deltaTone(dev?.heldout_delta)],
      ]),
    );
    if (curve?.status === "loading") return wrap.append(empty("Loading replay curve…"));
    if (!curve || curve.status !== "ready" || !curve.text) {
      return wrap.append(empty("No replay curve recorded.", curve?.error || "scaling_curve.json is missing."));
    }
    if (!parsedCurve) {
      wrap.append(el("p", { class: "pol-note" }, "Curve is not valid JSON."));
      wrap.append(codeBlock(curve.text, "text"));
      return;
    }
    const meta = curveObj ? Object.entries(curveObj).filter(([, value]) => value == null || typeof value !== "object") : [];
    if (meta.length) wrap.append(metaRow(meta));
    const tables = tablesFrom(parsedCurve);
    for (const table of tables) wrap.append(tableView(table.title, table.rows));
    if (!tables.length) wrap.append(el("p", { class: "pol-note" }, "No tabular rows in the curve."));
    wrap.append(rawBlock(curve.text, "Raw curve JSON"));
  }
  render(ctx.store.getState());
  return { update: render, destroy() {} };
}

/* --- verdict ------------------------------------------------------------- */

function factsNode(rows) {
  const dl = el("dl", { class: "pol-facts" });
  for (const [key, value] of rows) {
    dl.append(el("dt", {}, key));
    const dd = el("dd", {});
    if (value instanceof Node) dd.append(value);
    else dd.textContent = value == null || value === "" ? "—" : String(value);
    dl.append(dd);
  }
  return dl;
}

export function mountVerdict(root, ctx) {
  const wrap = el("div", { class: "pol pol-verdict" });
  root.append(wrap);
  let sig = "";
  function render(state) {
    const source = sel.activeSource(state);
    const step = source ? sel.stepForSource(state, source) : null;
    const entry = source ? sel.artifact(state, source, "verdict") : null;
    const next = `${source}|${entry?.status}|${entry?.at}|${step?.status}|${step?.mean_v}|${step?.delta_vs_best}|${step?.reason}`;
    if (next === sig) return;
    sig = next;
    clear(wrap);
    if (!source) return wrap.append(empty("Select a policy step."));
    const parsed = entry?.status === "ready" ? parseJson(entry.text) : null;
    const verdict = isObj(parsed) ? parsed : null;
    const status = verdict?.status || step?.status || null;
    wrap.append(viewHead(source, step, status));
    if (entry?.status === "loading") return wrap.append(empty("Loading verdict…"));
    if (!step && !verdict && entry?.status !== "ready") {
      return wrap.append(empty("No verdict yet.", entry?.error || "verdict.json is missing."));
    }
    wrap.append(
      factsNode([
        ["status", status ? pill(status) : "—"],
        ["slot", verdict?.slot ?? "—"],
        ["mean V", fmt.num(verdict?.mean_v ?? step?.mean_v)],
        ["delta vs best", fmt.signed(verdict?.delta_vs_best ?? step?.delta_vs_best)],
        ["attempts", verdict?.attempts ?? "—"],
        ["step", verdict?.step ?? step?.step ?? "—"],
      ]),
    );
    const error = verdict?.error || (entry?.status === "error" ? entry.error : null);
    if (error) wrap.append(el("p", { class: "pol-error" }, error));
    if (step?.reason) wrap.append(el("p", { class: "pol-note" }, [el("strong", {}, "Reason: "), step.reason]));
    if (entry?.status === "ready" && entry.text) {
      if (verdict) wrap.append(rawBlock(entry.text, "Raw verdict JSON"));
      else {
        wrap.append(el("p", { class: "pol-note" }, "Verdict is not JSON."));
        wrap.append(codeBlock(entry.text, "text"));
      }
    } else if (entry?.status === "error") {
      wrap.append(el("p", { class: "pol-note" }, `verdict.json not recorded: ${entry.error}`));
    }
  }
  render(ctx.store.getState());
  return { update: render, destroy() {} };
}
