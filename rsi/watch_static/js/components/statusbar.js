/* Status bar: transport, loop, GPU slots, last event, key hints. */

import { el, clear, setText, modLabel } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

export function mount(root, ctx) {
  const connection = el("span", { class: "status-item" }, [
    el("span", { class: "dot" }),
    el("span", { class: "text" }),
  ]);
  const loopItem = el("span", { class: "status-item" });
  const slots = el("span", { class: "status-item", dataset: { narrow: "hide" } });
  const last = el("span", { class: "status-item" });
  const hints = el("span", { class: "status-hints" }, [
    el("span", { class: "status-hint" }, [el("kbd", { class: "kbd" }, `${modLabel()} K`), "palette"]),
    el("span", { class: "status-hint" }, [el("kbd", { class: "kbd" }, "/"), "search"]),
  ]);
  root.append(connection, loopItem, slots, last, el("span", { class: "status-spacer" }), hints);

  function render(state) {
    const transport = state.transport;
    const tone =
      transport.status === "open" && !transport.frozen
        ? "live"
        : transport.status === "error"
          ? "err"
          : transport.status === "gone"
            ? "warn"
            : "idle";
    connection.dataset.tone = tone;
    connection.querySelector(".dot").dataset.tone = tone;
    setText(connection.querySelector(".text"), sel.connectionLabel(state));
    connection.title = transport.error || transport.reason || "";

    const loop = sel.loop(state);
    setText(loopItem, sel.statusLine(state));
    loopItem.dataset.tone = loop?.status === "error" ? "err" : loop?.status === "running" ? "live" : "idle";

    const gpuSlots = state.server.data?.protocol?.gpu_slots;
    const gpu = state.server.data?.gpu;
    setText(
      slots,
      gpuSlots == null ? "slots —" : `slots ${gpuSlots}${gpu ? ` · ${gpu.utilization_pct}%` : ""}`,
    );
    slots.title = gpu ? `${gpu.name}: ${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB` : "";

    const at = transport.lastEventAt;
    setText(last, at ? `event ${fmt.ago(at)}` : "no events");
    last.title = at ? new Date(at).toLocaleString() : "";
  }

  render(ctx.store.getState());
  return { update: render, destroy() {} };
}
