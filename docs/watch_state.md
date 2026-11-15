# Watch page state machines

The page has one store (`rsi/watch_static/js/store.js`) and one pure reducer
(`rsi/watch_static/js/reducer.js`). Server state, SSE transport state and UI state are separate
slices; a view never infers one from another. This file is the contract the reducer and the
node tests enforce.

## Slices

| Slice | Holds | Written by |
|---|---|---|
| `server` | last `/api/state` payload, its status, receipt time, sequence | `state/loading`, `state/loaded`, `state/error` |
| `transport` | the one SSE connection: source, status, cursor, done reason, `frozen` | `transport/*` |
| `transcript` | events for the open source, next index, load status | `transcript/*`, `transport/events`, `transport/reset` |
| `artifacts` | `"<source>|<name>" → {status, text, at, error}` | `artifact/*` |
| `context` | `"<source>" → {status, payload, at, error}` | `context/*` |
| `ui` | cycle, tabs, active tab, theme, panels, filter, search, palette, follow | `ui/*` |

## Machine 1 — loop (derived, read-only)

```
idle ──state/loaded(status=running, running=true)──▶ running
running ──state/loaded(status=stopped)──▶ stopped
running ──state/loaded(status=done)─────▶ done
running ──state/loaded(status=error)────▶ error
```

`loopLive = loop.running === true && loop.status === "running"`.
`loopEnded = status ∈ {stopped, done, error}`.
The server sets `running` from the study lock, so it is the only trustworthy liveness signal.

## Machine 2 — node phase (derived, never stored)

```
pending ──(agent starts)──▶ agent ──(eval dir appears)──▶ evaluating ──(score.json)──▶ done
   any of agent|evaluating ──(!loopLive)──▶ stopped
   done, pending, stopped are terminal
```

`nodePhase(state, node)` is the only place that decides this. A node on disk that says `agent`
while no process holds the lock is `stopped`; a stopped loop never leaves a node live.

## Machine 3 — transport (one connection at a time)

```
idle ──attach(source)──▶ connecting ──hello──▶ open
open ──events──▶ open            (cursor advances, events append)
open ──reset──▶ open             (events cleared, cursor 0)
open ──done(reason)──▶ closed|gone   (frozen = true)
open ──error──▶ error ──hello──▶ open     (bounded retries live in sse.js)
open|error ──detach──▶ idle      (frozen = false, events kept)
```

- `done` sets `frozen: true`; `closed` sets `frozen: true`.
- `hello` never clears `frozen`. A reconnect after a done frame must not resurrect a transcript.
- `state/loaded` with `loopEnded` freezes an open transport (`status: "closed"`).
- `frozen` is cleared only by `attach` (a new source) or `detach` (leaving the view).
- `transport/events` ignores frames for another source and frames that arrive while frozen.

## Machine 4 — transcript

```
idle ──transcript/loading──▶ loading ──transcript/loaded──▶ ready
ready ──transport/events──▶ ready (append)
ready ──transport/reset──▶ ready (clear, cursor 0)
loading|ready ──transcript/error──▶ error
```

Events are append-only; `reset` is the only clear, and it is ignored while frozen.
The initial `/api/events` load and the SSE stream share the transcript slice: the load sets it,
the stream appends to it.

## Machine 5 — editor tabs

```
none ──ui/open(source)──▶ one tab, active
tabs ──ui/open(source)──▶ focus existing, or append (max 12, oldest inactive dropped)
tabs ──ui/close(id)──▶ remove; if it was active, activate the next or previous tab
tabs ──ui/activate(id)──▶ active
active tab ──ui/view(v)──▶ view changes for that tab only
```

`tab.id === source` (`node:<cycle>:<cell>` or `policy:<cycle>:<step>`). Each tab remembers its
own view, so switching back restores it. `activeId` is `null` or an existing tab id.

## Machine 6 — theme

```
system ──ui/theme(light|dark)──▶ explicit ──ui/theme(system)──▶ system
```

`theme-boot.js` resolves the persisted choice before first paint. `system` re-resolves on
`prefers-color-scheme` changes. The choice is persisted under `watch.theme`.

## Invariants

1. `loopLive` requires both `running === true` and `status === "running"`.
2. `nodePhase` demotes `agent`/`evaluating` to `stopped` whenever `!loopLive`.
3. `frozen` is monotonic within a source: only `attach`/`detach` clear it.
4. `hello` never clears `frozen` and never clears events.
5. `transport/events` appends only for the attached source and only when not frozen.
6. `transport/reset` is ignored while frozen.
7. The reducer is pure: no `Date.now`, `Math.random`, `fetch`, DOM or storage. Timestamps arrive
   in actions.
8. Selectors are pure and DOM-free; every derived value (phase, rows, blocks, scores) has one
   selector.
9. One writer per DOM region: each module owns its root element and never touches another's.
10. Artifact and context results never overwrite a newer result (`at` guard).
11. A node's transcript is final once `done`/`gone`, even if the loop later restarts.
12. `?demo=1` uses the same store and reducer; only `api`/`sse` are swapped.

## Where the rules live

- `js/reducer.js`: transitions 3–6 and the `at` guards.
- `js/selectors.js`: machines 1–2, tree rows, transcript blocks, scores, routing helpers.
- `js/main.js`: effects (what to load for the active tab), polling, stream lifecycle, hash routing.
- `js/sse.js`: reconnect/backoff and frame parsing; it never dispatches directly.
- tests: `tests/test_rsi_watch_store.py` runs the reducer and selectors under node.
