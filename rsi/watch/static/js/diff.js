/* Patch parsing and line diffing. Pure: no DOM, importable under node.
 *
 * parsePatch(text)  -> {files:[{path, header, hunks:[{header, lines:[{kind, text, oldNo, newNo}]}]}]}
 * diffLines(a, b)   -> one file object of the same shape (line LCS, 3 lines of context)
 * fileStats(file)   -> {adds, dels}
 *
 * Malformed input is tolerated: whatever parsed is returned. A patch with no hunks
 * (a rename, a mode change) is kept so the file list can still show it.
 */

const MAX_LINES = 4000; // per side; beyond this the middle is replaced wholesale
const MAX_CELLS = 4_000_000; // LCS table budget (lines x lines)
const CONTEXT = 3;

const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;

export function parsePatch(text) {
  const files = [];
  let file = null;
  let hunk = null;
  let oldNo = 0;
  let newNo = 0;
  let fresh = false; // the last line was `diff --git`, so `---` belongs to this file

  const startFile = (path) => {
    file = { path: path || "", header: [], hunks: [] };
    files.push(file);
    hunk = null;
    oldNo = 0;
    newNo = 0;
  };

  const lines = String(text ?? "").split("\n");
  for (let index = 0; index < lines.length; index++) {
    const raw = lines[index];
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;

    if (line.startsWith("diff --git ")) {
      startFile(gitPath(line));
      file.header.push(line);
      fresh = true;
      continue;
    }
    if (line.startsWith("--- ")) {
      if (!file || (!fresh && (file.header.length || file.hunks.length))) startFile("");
      fresh = false;
      const path = cleanPath(line.slice(4));
      if (path) file.path = path;
      file.header.push(line);
      hunk = null;
      continue;
    }
    if (line.startsWith("+++ ")) {
      if (!file) startFile("");
      fresh = false;
      const path = cleanPath(line.slice(4));
      if (path) file.path = path;
      file.header.push(line);
      hunk = null;
      continue;
    }
    const match = HUNK_RE.exec(line);
    if (match) {
      if (!file) startFile("");
      fresh = false;
      oldNo = Number(match[1]);
      newNo = Number(match[3]);
      hunk = { header: line, lines: [] };
      file.hunks.push(hunk);
      continue;
    }
    if (!hunk) {
      // index/mode/similarity lines between the file header and the first hunk
      if (file && line && !line.startsWith("\\")) file.header.push(line);
      continue;
    }
    if (line === "" && index === lines.length - 1) continue; // trailing newline
    if (line.startsWith("\\")) {
      hunk.lines.push({ kind: "meta", text: line, oldNo: null, newNo: null });
      continue;
    }
    const marker = line[0];
    if (marker === "+") {
      hunk.lines.push({ kind: "add", text: line.slice(1), oldNo: null, newNo: newNo++ });
    } else if (marker === "-") {
      hunk.lines.push({ kind: "del", text: line.slice(1), oldNo: oldNo++, newNo: null });
    } else if (marker === " ") {
      hunk.lines.push({ kind: "ctx", text: line.slice(1), oldNo: oldNo++, newNo: newNo++ });
    } else if (line === "") {
      // an empty context line that lost its leading space
      hunk.lines.push({ kind: "ctx", text: "", oldNo: oldNo++, newNo: newNo++ });
    } else {
      hunk.lines.push({ kind: "meta", text: line, oldNo: null, newNo: null });
    }
  }

  for (const entry of files) {
    entry.path = entry.path || "patch";
    entry.header = entry.header.join("\n");
  }
  return { files };
}

export function diffLines(oldText, newText, options = {}) {
  const path = options.path || "candidate.py";
  const before = splitLines(oldText);
  const after = splitLines(newText);

  if (before.length > MAX_LINES || after.length > MAX_LINES) {
    return fileOf(path, [...before.map(del), ...after.map(add)]);
  }

  let head = 0;
  while (head < before.length && head < after.length && before[head] === after[head]) head++;
  let tail = 0;
  while (
    tail < before.length - head &&
    tail < after.length - head &&
    before[before.length - 1 - tail] === after[after.length - 1 - tail]
  ) {
    tail++;
  }

  const ops = [];
  for (let i = 0; i < head; i++) ops.push(ctx(before[i]));
  ops.push(...middle(before.slice(head, before.length - tail), after.slice(head, after.length - tail)));
  for (let i = before.length - tail; i < before.length; i++) ops.push(ctx(before[i]));
  return fileOf(path, ops);
}

export function fileStats(file) {
  let adds = 0;
  let dels = 0;
  for (const hunk of file?.hunks || []) {
    for (const line of hunk.lines) {
      if (line.kind === "add") adds++;
      else if (line.kind === "del") dels++;
    }
  }
  return { adds, dels };
}

/* --- helpers ------------------------------------------------------------- */

function splitLines(text) {
  const source = String(text ?? "");
  if (!source) return [];
  const lines = source.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();
  return lines.map((line) => (line.endsWith("\r") ? line.slice(0, -1) : line));
}

function middle(before, after) {
  if (!before.length) return after.map(add);
  if (!after.length) return before.map(del);
  if (before.length * after.length > MAX_CELLS) return [...before.map(del), ...after.map(add)];
  return lcs(before, after);
}

function lcs(before, after) {
  const n = before.length;
  const m = after.length;
  const width = m + 1;
  const table = new Uint32Array((n + 1) * width);
  for (let i = n - 1; i >= 0; i--) {
    const row = i * width;
    const next = row + width;
    for (let j = m - 1; j >= 0; j--) {
      table[row + j] =
        before[i] === after[j]
          ? table[next + j + 1] + 1
          : Math.max(table[next + j], table[row + j + 1]);
    }
  }
  const raw = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (before[i] === after[j]) {
      raw.push(ctx(before[i]));
      i++;
      j++;
    } else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) {
      raw.push(del(before[i]));
      i++;
    } else {
      raw.push(add(after[j]));
      j++;
    }
  }
  while (i < n) raw.push(del(before[i++]));
  while (j < m) raw.push(add(after[j++]));
  return groupRuns(raw);
}

/* Within a changed run, deletions come before additions, as in a unified diff. */
function groupRuns(ops) {
  const out = [];
  let dels = [];
  let adds = [];
  const flush = () => {
    out.push(...dels, ...adds);
    dels = [];
    adds = [];
  };
  for (const op of ops) {
    if (op.kind === "ctx") {
      flush();
      out.push(op);
    } else if (op.kind === "del") {
      dels.push(op);
    } else {
      adds.push(op);
    }
  }
  flush();
  return out;
}

function fileOf(path, ops) {
  return { path, header: "", hunks: buildHunks(numbered(ops)) };
}

function numbered(ops) {
  let oldNo = 1;
  let newNo = 1;
  return ops.map((op) => {
    const line = { kind: op.kind, text: op.text, oldNo: null, newNo: null };
    if (op.kind === "ctx") {
      line.oldNo = oldNo++;
      line.newNo = newNo++;
    } else if (op.kind === "del") {
      line.oldNo = oldNo++;
    } else if (op.kind === "add") {
      line.newNo = newNo++;
    }
    return line;
  });
}

function buildHunks(lines, context = CONTEXT) {
  const changed = [];
  for (let i = 0; i < lines.length; i++) if (lines[i].kind !== "ctx") changed.push(i);
  if (!changed.length) return [];

  const ranges = [];
  let start = Math.max(0, changed[0] - context);
  let end = Math.min(lines.length - 1, changed[0] + context);
  for (let k = 1; k < changed.length; k++) {
    const index = changed[k];
    if (index - context <= end + 1) {
      end = Math.min(lines.length - 1, index + context);
    } else {
      ranges.push([start, end]);
      start = Math.max(0, index - context);
      end = Math.min(lines.length - 1, index + context);
    }
  }
  ranges.push([start, end]);

  return ranges.map(([from, to]) => {
    const slice = lines.slice(from, to + 1);
    let oldStart = 0;
    let newStart = 0;
    let oldCount = 0;
    let newCount = 0;
    for (const line of slice) {
      if (line.oldNo != null) {
        if (!oldCount) oldStart = line.oldNo;
        oldCount++;
      }
      if (line.newNo != null) {
        if (!newCount) newStart = line.newNo;
        newCount++;
      }
    }
    if (!oldCount) oldStart = from === 0 ? 0 : (lines[from - 1].oldNo ?? 0);
    if (!newCount) newStart = from === 0 ? 0 : (lines[from - 1].newNo ?? 0);
    return { header: `@@ -${oldStart},${oldCount} +${newStart},${newCount} @@`, lines: slice };
  });
}

const ctx = (text) => ({ kind: "ctx", text });
const add = (text) => ({ kind: "add", text });
const del = (text) => ({ kind: "del", text });

function gitPath(line) {
  const rest = line.slice("diff --git ".length);
  const at = rest.indexOf(" b/");
  if (at > 0) return cleanPath(rest.slice(at + 1));
  return cleanPath(rest.split(/\s+/).pop());
}

function cleanPath(raw) {
  let path = String(raw || "").trim();
  if (!path) return null;
  const tab = path.indexOf("\t");
  if (tab >= 0) path = path.slice(0, tab);
  if (path.length > 1 && path.startsWith('"') && path.endsWith('"')) path = path.slice(1, -1);
  if (path === "/dev/null") return null;
  if (path.startsWith("a/") || path.startsWith("b/")) path = path.slice(2);
  return path || null;
}
