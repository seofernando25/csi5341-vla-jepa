# Two-computer presenter view

The slide computer runs a small Python relay and serves both screens. No account, cloud service, API key, npm install or software on the notes computer is needed. Use Python 3.10+ and a current browser. Both computers must be able to reach each other over the LAN.

## Start on the slide computer

From the repository root:

```sh
git pull --ff-only
python3 web/presentation/tools/relay.py --host 0.0.0.0 --port 8766
```

Keep this terminal running. It prints two URLs:

- **Slides:** `http://localhost:8766/?relay=control&key=<temporary-pairing-key>` — open the exact printed URL on the slide computer. The key pairs the controller; the browser removes it from the address bar and retains it for reloads in that tab. Only one controller can publish at a time. If replacing the controller tab, close the old tab and allow seven seconds for its lease to expire.
- **Notes:** `http://<slide-computer-LAN-IP>:8766/presenter.html` — open the exact printed Notes URL on the second computer. Do **not** use `localhost` there: that points to the second computer itself.

On a Mac that advertises a `.local` hostname, the relay also prints an mDNS alternative such as `http://Fernandos-MacBook-Air.local:8766/presenter.html`. This Mac currently uses that name. mDNS support varies by the second computer and network; use the IP URL if the name does not resolve.

The IP printed by the relay is its best routing estimate. If the computer has a VPN or multiple interfaces, use the address of its Wi-Fi interface instead (on a Mac, System Settings → Wi-Fi → Details → TCP/IP). For example, `http://192.168.1.25:8766/presenter.html`. Use the same port on both URLs if you change `--port`.

Use the existing controls on the slide computer: Right/Space/click advances, Left/right-click goes back, Replay beat repeats, Chapters jumps, and N opens manual notes. Right during active motion still accelerates it. The notes screen has no slide navigation. Its text-size control (24–64 px) is remembered by that browser; Full screen enlarges the reading area. The chapter/cue header stays fixed. Only the script pane scrolls manually (wheel, trackpad, or focus it and use Up/Down/Page Down). New chapters return the script to the top; cue changes preserve its reading position. The surrounding page never scrolls.

Notes map to **whole chapters**. The cue badge identifies the current animation beat and whether it is playing or holding; there is no word highlighting, timed auto-scroll or claimed word-level synchronization. Delivery guidance can be expanded separately.

## Connections and recovery

The viewer receives the current chapter/cue immediately on connect, including when joining halfway through. It reconnects automatically after a network interruption and receives the latest snapshot, rather than playing missed updates. The slide tab sends updates and a two-second heartbeat. A disconnected slide computer is reported after about seven seconds; the last script remains visible with a stale-state warning. Network failures do not block local advance/back/replay.

If you restart the relay process, it forgets its state and generates a new pairing key. Open the **newly printed Slides URL** in the existing slide tab to pair again, then select the desired chapter. The notes browser can reconnect at its unchanged URL. If the host IP or port changes, update its URL too. Do not open the public HTTPS deployment for this workflow: both screens need these local HTTP relay URLs.

If the notes page cannot connect:

1. Confirm the relay terminal is still running, and use its host IP and port on the second computer.
2. Allow incoming Python connections on the slide computer’s firewall for the local network. No port forwarding is required.
3. Guest/campus Wi-Fi may use **client/AP isolation**, blocking communication even on the same SSID. Use a private router, a hotspot that permits clients to communicate, or a wired LAN. A shared Wi-Fi name alone does not guarantee connectivity.
4. If the network cannot support the relay, keep the existing **N / Notes** view on the slide computer. Ordinary static hosting remains unchanged and requires no relay.

The relay holds navigation state only in memory. Its temporary key authorizes publishing; the read-only notes URL is accessible to devices that can reach this local port. It serves presentation runtime assets only, not the repository or authoring export endpoints. Stop it with Ctrl+C when finished.

## Code and checks

- `tools/relay.py`: local HTTP server, state validation, controller lease and server-sent events (SSE).
- `src/sync/controller.js`: optional adapter, enabled only by `?relay=control`. Receives state notifications from `player.js`; existing navigation remains independent.
- `presenter.html`, `presenter.css`, `src/sync/presenter.js`: read-only chapter notes, connection status, reconnect handling and reading preferences. Share the existing `content.js` and named `timing/cues.js` definitions.

```sh
python3 web/presentation/tools/tests/relay_test.py
node web/presentation/tools/tests/manual-cues.cjs
node web/presentation/tools/tests/arrowheads.cjs
```

The relay regression opens real HTTP/SSE connections to check initial/late snapshots, forward/back state, replay motion status, reconnection after missed navigation, invalid state rejection, controller ownership and expired-controller recovery. Clock regression covers all 18 chapters and 48 beats. Browser smoke checks cover navigation, viewer reload/reconnect, font size and manual notes. A real second-device rehearsal is still needed to establish that your Wi-Fi/firewall permits peer connections.
