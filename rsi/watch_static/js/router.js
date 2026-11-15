/* Hash routing: #/node/<cycle>/<cell>/<view> and #/policy/<cycle>/<step>/<view>. */

import { isView } from "./views/meta.js";

export function parseHash(hash = location.hash) {
  const text = String(hash).replace(/^#\/?/, "");
  if (!text) return null;
  const parts = text.split("/");
  if (parts[0] !== "node" && parts[0] !== "policy") return null;
  if (parts.length < 3) return null;
  const [kind, cycle, target, view] = parts;
  if (!cycle || !target) return null;
  return {
    kind,
    cycle: decodeURIComponent(cycle),
    target: decodeURIComponent(target),
    view: view && isView(kind, view) ? view : null,
  };
}

export function formatHash(tab) {
  if (!tab) return "";
  return `#/${tab.kind}/${encodeURIComponent(tab.cycle)}/${encodeURIComponent(tab.target)}/${tab.view}`;
}

export function readHash() {
  return parseHash(location.hash);
}

export function writeHash(tab, { replace = false } = {}) {
  const next = formatHash(tab);
  if (!next || location.hash === next) return;
  if (replace) history.replaceState(null, "", next);
  else history.pushState(null, "", next);
}
