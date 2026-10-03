# Two-computer presenter view

The slide computer runs a small Python relay and serves both screens. No account, cloud service, API key, npm install or software on the notes computer is needed. Use Python 3.10+ and a current browser. Both computers must be able to reach each other over the LAN.

## Start on the slide computer

From the repository root:

```sh
git pull --ff-only
python3 web/presentation/tools/relay.py --host 0.0.0.0 --port 8766
```

Keep this terminal running. It prints two URLs:

- **Controller:** `http://localhost:8766/?relay=control` on the slide computer, or `http://<slide-computer-LAN-IP>:8766/?relay=control`. The relay pairs this route automatically; no key copying is needed. Only one controller can publish at a time. Close the old controller and allow seven seconds for its lease to expire before replacing it.
- **Slide viewer:** `http://<slide-computer-LAN-IP>:8766/` on every other slide screen. It follows chapter/cue animations automatically and cannot advance, go back, replay or jump locally. Notes and full screen still work. A late/reconnected viewer receives the latest state. Looping illustration videos retain their own playback clock.
- **Notes:** `http://<slide-computer-LAN-IP>:8766/presenter.html` — open the exact printed Notes URL on the second computer. Do **not** use `localhost` there: that points to the second computer itself.

On a Mac that advertises a `.local` hostname, the relay also prints an mDNS alternative such as `http://Fernandos-MacBook-Air.local:8766/presenter.html`. This Mac currently uses that name. mDNS support varies by the second computer and network; use the IP URL if the name does not resolve.

The IP printed by the relay is its best routing estimate. If the computer has a VPN or multiple interfaces, use the address of its Wi-Fi interface instead (on a Mac, System Settings → Wi-Fi → Details → TCP/IP). For example, `http://192.168.1.25:8766/presenter.html`. Use the same port on both URLs if you change `--port`.

Use the existing controls on the slide computer: Right/Space/click advances, Left/right-click goes back, Replay beat repeats, Chapters jumps, and N opens manual notes. Right during active motion still accelerates it. The notes screen has no slide navigation. Its text-size control (24–64 px) is remembered by that browser; Full screen enlarges the reading area. Chapter, cue and connection status stay in the bottom dock, leaving the top for the script. Only the script pane moves; you can also scroll it manually (wheel, trackpad, or focus it and use Up/Down/Page Down). New chapters return the script to the top; cue changes preserve its reading position. The surrounding page never scrolls. Automatic upward scrolling starts after a short reading lead-in; adjust Global speed (pixels/second), Pause/Resume, or Back to top in the bottom toolbar. Wheel/touch/manual reading pauses automatic motion. Each new chapter starts at the top; scroll stops with its final script line at the top. Connection loss pauses motion until the slide computer returns. Global speed is shared: anyone on the controller, slide viewer or presenter page can change it. All connected screens receive the latest pace, including after reconnecting. Pause/Resume, reading position and text size remain local. Scrolling is independent of narration and cues; it is not word-level synchronization.

A large live picture-in-picture (38% of the viewport width, up to 720 px) sits in the top-right corner beside the notes, on a plain white reading surface. It mirrors one offscreen slide renderer and follows the same relay chapter/cue state, including reconnects. **Hide preview / Show preview** in the bottom dock toggles it locally; this preference is remembered. The embedded renderer has no controls and cannot navigate.

Notes map to **whole chapters**. The cue badge identifies the current animation beat and whether it is playing or holding; automatic scrolling uses your selected pace, with no word highlighting or claimed word-level synchronization. Delivery guidance can be expanded separately.

## Connections and recovery

The viewer receives the current chapter/cue immediately on connect, including when joining halfway through. It reconnects automatically after a network interruption and receives the latest snapshot, rather than playing missed updates. The slide tab sends updates and a two-second heartbeat. A disconnected slide computer is reported after about seven seconds; the last script remains visible with a stale-state warning. Network failures do not block local advance/back/replay.

If you restart the relay process, it forgets its state and resets global speed to 10 px/s. Reload the controller route to pair automatically, then select the desired chapter. The notes browser can reconnect at its unchanged URL. If the host IP or port changes, update its URL too. Do not open the public HTTPS deployment for this workflow: both screens need these local HTTP relay URLs.

If the notes page cannot connect:

1. Confirm the relay terminal is still running, and use its host IP and port on the second computer.
2. Allow incoming Python connections on the slide computer’s firewall for the local network. No port forwarding is required.
3. Guest/campus Wi-Fi may use **client/AP isolation**, blocking communication even on the same SSID. Use a private router, a hotspot that permits clients to communicate, or a wired LAN. A shared Wi-Fi name alone does not guarantee connectivity.
4. If the network cannot support the relay, keep the existing **N / Notes** view on the slide computer. Ordinary static hosting remains unchanged and requires no relay.

The relay holds navigation state and shared speed only in memory. The control route receives an ephemeral publishing key automatically; it is a convenience boundary on a trusted LAN, not an account/password access restriction. Share the plain viewer URL with viewers; the read-only notes URL is accessible to devices that can reach this local port. It serves presentation runtime assets only, not the repository or authoring export endpoints. Stop it with Ctrl+C when finished.

## Code and checks

- `tools/relay.py`: local HTTP server, state validation, controller lease and server-sent events (SSE).
- `src/sync/viewer.js`: read-only relay slide follower, including direction/progress snapshots. Normal public/static hosting retains standalone controls.
- `src/sync/speed-sync.js`: shared pace controls, validated settings writes and revision-aware SSE/reconnect updates on all three routes.
- `src/sync/auto-scroll.js`: smooth elapsed-time reading motion, pause, speed, manual override and final-line hold.
- `src/sync/controller.js`: optional adapter, enabled only by `?relay=control`. Receives state notifications from `player.js`; existing navigation remains independent.
- `presenter.html`, `presenter.css`, `src/sync/presenter.js`: read-only chapter notes, connection status, reconnect handling and reading preferences. Share the existing `content.js` and named `timing/cues.js` definitions.

```sh
python3 web/presentation/tools/tests/relay_test.py
node web/presentation/tools/tests/manual-cues.cjs
node web/presentation/tools/tests/arrowheads.cjs
node web/presentation/tools/tests/auto-scroll.cjs
node web/presentation/tools/tests/viewer-sync.cjs
```

The relay regression opens real HTTP/SSE connections to check initial/late snapshots, forward/back state, replay motion status, reconnection after missed navigation, invalid state rejection, controller ownership and expired-controller recovery. Clock regression covers all 18 chapters and 48 beats. Browser smoke checks cover navigation, viewer reload/reconnect, font size and manual notes. A real second-device rehearsal is still needed to establish that your Wi-Fi/firewall permits peer connections.

## Speaking split

The script alternates in six sections: Noah 1–2, Fernando 3–4, Noah 5–8, Fernando 9–12, Noah 13–16, Fernando 17–18. Each speaker has exactly 615 scripted words. At the same speaking pace, each has about 4 minutes 40 seconds of speech, including short paragraph breaths. The remaining time within ten minutes is for visual holds and handovers. This is a script-based estimate, not a measured rehearsal. The footer shows the assigned speaker, and delivery guidance indicates handovers.
