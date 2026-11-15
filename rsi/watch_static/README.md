# Watch page

Read-only, loopback-only workbench for one Dream-RSI study. Served by
`vla-rsi watch --study-dir DIR [--port 8787]`. No build step, no dependencies, no network: the
page is a tree of ES modules and plain CSS under this directory.

```sh
uv run --no-sync vla-rsi watch --study-dir studies/program-v1
# offline demo with sanitized fixtures:
open http://127.0.0.1:8787/?demo=1
# demo scenarios: ?demo=1&scenario=stopped | error | empty
# deterministic screenshots: ?demo=1&still=1 (one batch instead of a scripted stream)
```

## Module map

| Path | Owns |
|---|---|
| `index.html` | shell markup: title bar, sidebar, editor, inspector, status bar, overlay roots |
| `css/tokens.css` | every colour, font, size and layout token (light default, `[data-theme="dark"]`) |
| `css/base.css` | reset, typography, scrollbars, focus ring, shared primitives (`.btn`, `.pill`, `.empty`…) |
| `css/shell.css` | app grid, title bar, sidebar, tabs, view bar, editor, inspector, status bar, overlays |
| `css/tree.css`, `transcript.css`, `outline.css`, `context.css`, `proposal.css`, `code.css`, `evaluation.css`, `inspector.css`, `policy.css`, `palette.css`, `responsive.css` | one module each |
| `js/theme-boot.js` | classic script in `<head>`: applies the persisted theme before first paint |
| `js/main.js` | bootstrap: store, polling, stream lifecycle, hash routing, layout, view mounting |
| `js/store.js` | `createStore(reducer, initial)`: `getState`, `dispatch`, `subscribe` |
| `js/reducer.js` | the only state transitions; pure, no DOM/clock/storage |
| `js/actions.js` | action creators |
| `js/selectors.js` | pure derivations: phases, tree rows, transcript blocks, scores, routing |
| `js/format.js` | pure formatters (`pct`, `dur`, `usd`, `chars`, `tokens`, labels) |
| `js/dom.js` | `el`, `svg`, `icon`, `clear`, `setText`, `copyButton`, `toast`, `rafBatch` |
| `js/api.js` | fetch client (`state`, `events`, `artifact`, `context`); swaps to `demo.js` |
| `js/sse.js` | EventSource wrapper: frame parsing, bounded reconnect, cursor |
| `js/router.js` | `#/node/<cycle>/<cell>/<view>`, `#/policy/<cycle>/<step>/<view>`, `#/story/<chapter>` |
| `js/keyboard.js` | global key map |
| `js/markdown.js` | markdown → DOM fragment (no `innerHTML`) |
| `js/highlight.js` | python / json / diff tokenizers → DOM |
| `js/diff.js` | unified-patch parser and line LCS |
| `js/views/meta.js` | view ids and labels per tab kind |
| `js/views/index.js` | registry: `mountView(kind, view, root, ctx)` |
| `js/views/tree.js` | explorer tree (sidebar) |
| `js/views/policy.js` | policy-development ledger (sidebar) |
| `js/views/transcript.js` | turn-structured transcript |
| `js/views/outline.js` | transcript outline, search, filter, follow |
| `js/views/context.js` | what the agent received |
| `js/views/proposal.js` | rendered `proposal.md` |
| `js/views/diff.js` | file list + unified diff |
| `js/views/solution.js` | line-numbered `solution.py` |
| `js/views/evaluation.js` | score, smoke, size, timings, fail class, error |
| `js/views/policy_step.js` | policy step prompt / diff / replay / verdict |
| `js/components/titlebar.js` | run name, status, cycle switcher, metrics, actions |
| `js/components/tabs.js` | editor tabs and the view switcher |
| `js/components/inspector.js` | selected node facts, lineage, chart |
| `js/components/chart.js` | dependency-free SVG score chart |
| `js/components/statusbar.js` | transport, loop, slots, last event, key hints |
| `js/components/palette.js` | command palette |
| `js/components/shortcuts.js` | shortcut sheet |
| `js/demo.js` | fixture adapter and scripted stream for `?demo=1` |
| `js/story/index.js` | Story shell: top bar, chapter rail, deep links, `ctx.story` data cache |
| `js/story/nav.js` | chapter metadata, story hash parsing, scroll spy |
| `js/story/model.js`, `model_dream.js` | pure Story derivations (baseline, probe, order, frontier, dream, diff) |
| `js/story/chapter-*.js` | one chapter each: task, probe, tree, dreaming, changed |
| `js/story/method.js` | loop steps and documented deviations |
| `js/story/dreaming-*.js` | dreaming figures and charts |
| `css/story.css`, `story-figures.css`, `story-<chapter>.css` | Story design system and per-chapter figures |
| `fixture.json`, `fixture_events.json`, `fixture_story.json` | sanitized demo state, events, artifacts, context, dream payload |
| `package.json` | `{"type":"module"}` so node parses the modules in tests |

## Data flow

```
/api/state (poll 2 s) ──┐
                        ├─▶ store.dispatch ──▶ reducer ──▶ subscribers ──▶ regions render
/api/stream (SSE) ──────┘                                   │
  hello/events/reset/done/state                             └─▶ effects(): load artifacts/context
                                                                    for the active tab
```

- `main.js` polls `/api/state` every 2 s. While a transcript is open the SSE `state` frames also
  update the server slice (sequence-guarded), so the title bar stays current.
- Opening a node in the transcript view loads `/api/events?after=0`, then tails `/api/stream`.
  `done`/`gone`/loop-end freeze the transcript; `hello` never unfreezes it.
- Artifacts and context are loaded on demand for the active view and cached by `source|name`.
- The URL hash is the source of truth for the open tab and view; every tab change is deep-linkable.

State machines, transitions and invariants: `docs/watch_state.md`.

## API (read-only, loopback only)

All responses are JSON except `/` and static assets. Every string is redacted by
`rsi.watch_events.redact_obj`. The server rejects a non-loopback `Host` (DNS rebinding) and any
method other than `GET`.

| Endpoint | Returns |
|---|---|
| `GET /` | `index.html` |
| `GET /css/<name>.css`, `/js/<name>.js`, `/js/views/<name>.js`, `/js/components/<name>.js` | static module (allowlist; no traversal, no symlinks) |
| `GET /fixture.json`, `/fixture_events.json` | demo fixtures |
| `GET /api/state` | `{study_dir, now, loop, protocol, gpu, cycles}` — see below |
| `GET /api/events?source=<src>&after=<n>` | `{events, next, done}`; `events` are watch events |
| `GET /api/stream?source=<src>&after=<n>` | SSE: `hello`, `events`, `reset`, `state`, `done`; `id:` is the event cursor |
| `GET /api/context?source=<src>` | `{source, kind, log, files, command, sections}` |
| `GET /api/artifact?source=<src>&name=<name>` | `{name, text}` or 404; names: `proposal`, `diff`, `score`, `error`, `solution`, `policy`, `seed`, `verdict`, `curve` |

`source` is `node:<cycle>:<cell>` or `policy:<cycle>:<step>`. Bad ids are 400, never a path.

`loop` = `{status, stop_reason, iteration, stage, cycles_target, elapsed_s, agent_cost_usd,
best_score, running, probe_budget}`. `running` is whether a process holds the study lock;
`probe_budget` is the run's `--probe-budget` (not the grid maximum).

`cycle` = `{cycle_id, iteration, status, decision_rounds, opened_branches, probes_used,
effective_sequential_rounds, agent_timeout_s, policy_sha256, nodes, policy_dev}`.

`node` = `{cell_id, parent_id, branch, attempt, round, status, fail_class, score, size_reduction,
successes, episodes, resident_mb, commit, agent_seconds, eval_seconds, agent_cost_usd,
agent_turns, agent_timed_out, error, headline, promoted, started_utc, elapsed_s}`.

`policy_dev` = `{status, selected_step, incumbent_v, selected_v, heldout_delta, steps}`; a step is
`{step, status, mean_v, delta_vs_best, reason}`.

`context.sections[]` = `{id, title, recorded, text, chars, tokens, path, note, items, fields}`.
`recorded: false` means the run does not write it (the agent CLI's system prompt). `items` are
earlier-node files (`cell_id` + `files`) or policy evidence files (`name` + `text`). `fields` is
the agent configuration from `logs/command.json`; the sandbox argv is deliberately not shipped.

## Add a view

1. Pick a tab kind (`node` or `policy`) and add an id/label to `js/views/meta.js`.
2. Create `js/views/<name>.js` exporting `mount(root, ctx)` → `{update, destroy}`.
3. Register it in `js/views/index.js`.
4. If it needs data, add the artifact names to `VIEW_ARTIFACTS` in `js/main.js` (or call
   `ensureContext` for `/api/context`).
5. Add `css/<name>.css` and a `<link>` in `index.html` (or reuse an existing file).
6. Add a fixture artifact/context entry so `?demo=1` covers it.
7. Test: `node --input-type=module --check < file.js`, then the pytest suites.

## Tests

```sh
uv run --no-sync pytest -q tests/test_rsi_watch.py tests/test_rsi_watch_events.py \
  tests/test_rsi_watch_static.py tests/test_rsi_watch_store.py
```

- `test_rsi_watch.py`: state builder, events, redaction, artifacts, context sections, server
  routes, static allowlist, host check.
- `test_rsi_watch_static.py`: offline/module rules, fixture contract, no markup injection, node
  parse checks.
- `test_rsi_watch_store.py`: reducer and selector invariants under node (skipped if node is
  absent).
- `test_rsi_watch_events.py`: transcript dialects and redaction (behaviour is frozen; the
  experiment engine imports `redact`, `pi_cost`, `pi_text`).

## Rules

- Read-only. Never add a mutating endpoint.
- Loopback only; the page has no authentication.
- No external origins, fonts, CDNs or build step.
- No `innerHTML` or markup strings: build nodes with `js/dom.js`.
- One writer per DOM region; views never touch another region.
- Never invent data: show `—` or "not recorded".

## Story mode

`#/story/<chapter>` is an editorial, scroll-driven explanation of the same run; the Workbench is
unchanged. The switch in both top bars is a plain link, so every chapter is deep-linkable and the
browser back button works. Chapters are mounted once by `js/story/index.js`; each exports
`{id, number, title, dek, mount(root, ctx)}` and returns `{update(state, prev), destroy()}`.
`ctx.story` caches `dream(cycle, world)`, `events(source)`, `artifact(source, name)` and
`context(source)`; failures resolve `null` and the chapter renders a state line, never invented
data. Scrubbers and play/pause are chapter-local ephemeral state; everything derived comes from
`js/story/model.js`, which is pure and node-tested (`tests/js/story_model.mjs`).

Story is light-first but uses the same tokens, so dark works. Motion is opt-in (play buttons) and
`prefers-reduced-motion` jumps to the final frame. No chapter uses `innerHTML`.

### Adding a chapter

1. Add `js/story/chapter-<id>.js` exporting `id`, `number`, `title`, `dek`, `mount`.
2. Add its metadata to `CHAPTERS` in `js/story/nav.js` (order = reading order).
3. Add `css/story-<id>.css` and link it in `index.html` (the static test requires every CSS file
   to be linked).
4. Import it in the `CHAPTER_MODULES` list in `js/story/index.js`.
5. If it needs new data, add a pure selector to `js/story/model.js` and a node assertion.

## Endpoints the page reads

| Endpoint | Payload |
|---|---|
| `GET /api/state` | `{study_dir, now, loop, protocol, gpu, cycles}`; `loop.baseline {score, resident_bytes}`; each node also carries `baseline_mb`, `per_task`, `peak_cuda_bytes`, `stage_seconds` when `score.json`/`eval/result.json` record them; `cycle.probes_used_live` counts scored nodes |
| `GET /api/events?source=&after=` | `{events, next, done}`; kinds `system, text, thinking, tool_call, tool_result, result`; thinking/text are streamed deltas |
| `GET /api/artifact?source=&name=` | `{name, text}`; names `proposal diff score error solution policy verdict curve seed export prompt` |
| `GET /api/context?source=` | `{source, kind, log, files, command, sections}`; `recorded:false` means not recorded |
| `GET /api/dream?cycle=&world=` | policy-development replay (below); `status: pending|ready|missing` |

`/api/dream` is read-only and returns `null` for anything the run did not write. `rounds` is
present only when `world` is given, and only for that world (all slots, all betas). Absolute paths
are never returned; `source_dir` is relative to the study or `null`.

```json
{"cycle":"iter_000","status":"pending|ready|missing","recorded_utc":null,
 "protocol":{"m":4,"lambda_parallel":0.05,"beta_grid":[0,0.25,0.5,0.75,1],"max_parallelism":2},
 "worlds":[{"id":"w0","nodes":12,"baseline_score":0.0,"max_gain":0.8,"degenerate":false}],
 "tree":{"world_id":"w0","nodes":[{"cell_id":"b0_a0","parent_id":null,"branch":0,"attempt":0,
   "seq":1,"score":0.08,"fail_class":"ok","evaluated":true,"summary":"..."}]},
 "candidates":[{"slot":"m0_incumbent","step":0,"status":"incumbent","mean_v":0.485,
   "delta_vs_best":null,"attempts":0,"error":null,"sha256":"..."}],
 "curves":{"m0_incumbent":{"w0":[{"beta":0.5,"probes":3,"attainment":0.4,"rounds":2}]}},
 "rounds":{"m0_incumbent":{"0":{"0.5":[{"round":1,"batch":["b0_a0"],"prefix_size":0,
   "outcomes":[{"cell_id":"b0_a0","score":0.08,"fail_class":"ok"}]}]}}},
 "heldout":{"num_worlds":24,"mean_V_by_slot":{},"delta_vs_incumbent":0.03},
 "verdict":{"selected_slot":"m3_llm","incumbent_in_sample_V":0.485,"selected_in_sample_V":0.579},
 "agent":{"backend":"pi","model":"...","effort":"max","calls":3,"num_turns":98,"cost_usd":0.52}}
```
