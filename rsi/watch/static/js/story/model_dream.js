/* Dream payload normalisation, replay frames, the changed-policy summary and a
 * line diff. Pure: no DOM, no clock, no imports. Re-exported by model.js. */

const MAX_DIFF_LINES = 500;

function list(value) {
  return Array.isArray(value) ? value : [];
}

function num(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function currentCycle(state) {
  const cycles = list(state?.server?.data?.cycles);
  const id = state?.ui?.cycleId;
  return cycles.find((cycle) => cycle?.cycle_id === id) || cycles[cycles.length - 1] || null;
}

/* ---------------------------------------------------------------------------
 * Dream payload
 * ------------------------------------------------------------------------- */

export function dreamModel(payload) {
  const data = payload && typeof payload === "object" ? payload : null;
  const worlds = list(data?.worlds);
  const candidates = list(data?.candidates);
  const curves = data?.curves && typeof data.curves === "object" ? data.curves : null;
  const rounds = data?.rounds && typeof data.rounds === "object" ? data.rounds : null;
  const tree = data?.tree && typeof data.tree === "object" ? data.tree : null;

  const world = (id) => {
    if (id == null) return worlds[0] || null;
    return worlds.find((item) => item?.id === id) || null;
  };
  /* The payload carries one world's tree; asking for another world is a miss. */
  const treeFor = (id) => {
    if (!tree) return null;
    if (id != null && tree.world_id != null && tree.world_id !== id) return null;
    return { ...tree, nodes: list(tree.nodes) };
  };
  const roundsFor = (slot, worldId, beta) => list(rounds?.[slot]?.[String(worldId)]?.[String(beta)]);

  return {
    pending: data?.status !== "ready",
    worlds,
    candidates,
    curves,
    verdict: data?.verdict || null,
    heldout: data?.heldout || null,
    agent: data?.agent || null,
    protocol: data?.protocol || null,
    world,
    tree: treeFor,
    rounds: roundsFor,
  };
}

/* Attainment recorded for this beta, at the last curve point already reached. */
function attainmentAt(curves, slot, worldId, beta, round) {
  if (beta == null) return null;
  const points = curves?.[slot]?.[String(worldId)];
  if (!Array.isArray(points)) return null;
  const wanted = Number(beta);
  let value = null;
  for (const point of points) {
    if (!point || Number(point.beta) !== wanted) continue;
    const at = point.rounds == null ? 0 : Number(point.rounds);
    if (Number.isFinite(at) && at <= round) value = point.attainment ?? null;
  }
  return value;
}

/* One frame of the replay: every candidate's batches up to `round` at one
 * beta, the cells they covered and the probes they spent. `last` is the
 * deepest round any candidate recorded, so the chapter knows where to stop. */
export function dreamFrame(payload, worldId, beta, round) {
  const model = dreamModel(payload);
  const prepared = model.candidates.map((candidate) => {
    const points = model
      .rounds(candidate?.slot, worldId, beta)
      .filter((point) => point && Number.isFinite(Number(point.round)))
      .map((point) => ({ ...point, round: Number(point.round) }))
      .sort((a, b) => a.round - b.round);
    return { slot: candidate?.slot ?? null, points };
  });
  const last = prepared.reduce((top, item) => {
    const deepest = item.points[item.points.length - 1]?.round ?? 0;
    return Math.max(top, deepest);
  }, 0);
  const wanted = Number(round);
  const frame = round == null || !Number.isFinite(wanted) ? last : Math.max(0, Math.min(wanted, last));

  const paths = prepared.map(({ slot, points }) => {
    const cells = [];
    let probes = 0;
    for (const point of points) {
      if (point.round > frame) continue;
      const batch = list(point.batch);
      cells.push(...batch);
      probes += Array.isArray(point.outcomes) ? point.outcomes.length : batch.length;
    }
    return { slot, cells, probes, attainment: attainmentAt(model.curves, slot, worldId, beta, frame) };
  });
  return { rounds: frame, last, paths };
}

/* ---------------------------------------------------------------------------
 * What changed
 * ------------------------------------------------------------------------- */

export function changedModel(state, payload) {
  const data = payload && typeof payload === "object" ? payload : null;
  const candidates = list(data?.candidates);
  const verdict = data?.verdict || null;
  const dev = currentCycle(state)?.policy_dev || null;
  const before = num(verdict?.incumbent_in_sample_V) ?? num(dev?.incumbent_v);
  const after = num(verdict?.selected_in_sample_V) ?? num(dev?.selected_v);
  return {
    incumbent: candidates.find((candidate) => candidate?.status === "incumbent") || null,
    selected: verdict?.selected_slot ?? null,
    delta: before != null && after != null ? after - before : null,
    heldoutDelta: num(data?.heldout?.delta_vs_incumbent),
    variants: candidates.map((candidate) => ({
      slot: candidate?.slot ?? null,
      step: num(candidate?.step),
      status: candidate?.status || null,
      meanV: num(candidate?.mean_v),
      deltaVsBest: num(candidate?.delta_vs_best),
      attempts: num(candidate?.attempts),
      error: candidate?.error || null,
    })),
  };
}

/* ---------------------------------------------------------------------------
 * Line diff
 * ------------------------------------------------------------------------- */

function lines(value) {
  const text = String(value ?? "").replace(/\r\n?/g, "\n");
  const trimmed = text.endsWith("\n") ? text.slice(0, -1) : text;
  if (!trimmed) return [];
  return trimmed.split("\n").slice(0, MAX_DIFF_LINES);
}

export function diffLines(before, after) {
  const a = lines(before);
  const b = lines(after);
  const n = a.length;
  const m = b.length;
  const width = m + 1;
  const table = new Uint32Array((n + 1) * width);
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[i * width + j] =
        a[i] === b[j]
          ? table[(i + 1) * width + j + 1] + 1
          : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
    }
  }
  const out = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m && out.length < MAX_DIFF_LINES) {
    if (a[i] === b[j]) {
      out.push({ kind: "same", text: a[i] });
      i += 1;
      j += 1;
    } else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) {
      out.push({ kind: "del", text: a[i] });
      i += 1;
    } else {
      out.push({ kind: "add", text: b[j] });
      j += 1;
    }
  }
  while (i < n && out.length < MAX_DIFF_LINES) out.push({ kind: "del", text: a[i++] });
  while (j < m && out.length < MAX_DIFF_LINES) out.push({ kind: "add", text: b[j++] });
  return out;
}
