/* Optional network adapter. Navigation/clock code never depends on the relay. */
(() => {
  const params = new URLSearchParams(location.search);
  if (params.get("relay") !== "control") return;
  const key = params.get("key");
  // Keep the pairing key out of copied URLs/history after initial setup.
  if (key) {
    sessionStorage.setItem("presenter-key", key);
    params.delete("key");
    history.replaceState(null, "", location.pathname + "?" + params);
  }
  const pairingKey = key || sessionStorage.getItem("presenter-key") || "";
  const client =
    sessionStorage.getItem("presenter-client") ||
    Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) =>
      b.toString(16).padStart(2, "0"),
    ).join("");
  sessionStorage.setItem("presenter-client", client);
  const status = document.createElement("span");
  status.id = "relay-status";
  status.setAttribute("role", "status");
  status.style.cssText = "font-size:13px;padding:0 8px;color:#62748a";
  status.textContent = "Presenter relay: connecting…";
  const link = document.createElement("a");
  link.href = "presenter.html";
  link.target = "_blank";
  link.textContent = "Presenter view";
  link.style.cssText = "font-size:14px;color:#0086be;padding:8px";
  document.getElementById("controls").append(link, status);
  let sequence = Number(sessionStorage.getItem("presenter-sequence")) || 0;
  let lastSent = "",
    busy = false,
    pending = false;
  function currentState() {
    return { scene: index, cue: presenter.cursor, running: presenter.running };
  }
  async function publish(force = false) {
    const state = currentState(),
      serialized = JSON.stringify(state);
    if (!force && serialized === lastSent) return;
    if (busy) {
      pending = true;
      return;
    }
    busy = true;
    sequence += 1;
    sessionStorage.setItem("presenter-sequence", sequence);
    try {
      const response = await fetch("relay/state", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Presenter-Key": pairingKey,
        },
        body: JSON.stringify({ client, sequence, state }),
        signal: AbortSignal.timeout(2500),
      });
      if (!response.ok)
        throw new Error(
          response.status === 409
            ? "Another controller is connected"
            : "Pairing failed",
        );
      lastSent = serialized;
      status.textContent = "Presenter relay: connected";
    } catch (error) {
      lastSent = "";
      status.textContent =
        error.message === "Another controller is connected" ||
        error.message === "Pairing failed"
          ? error.message
          : "Presenter relay: disconnected · retrying";
    } finally {
      busy = false;
      if (pending) {
        pending = false;
        publish();
      }
    }
  }
  window.publishPresenterState = () => publish();
  window.addEventListener("online", () => publish(true));
  setInterval(() => publish(true), 2000); // Also renews controller lease/re-publishes after a relay restart.
  publish(true);
})();
