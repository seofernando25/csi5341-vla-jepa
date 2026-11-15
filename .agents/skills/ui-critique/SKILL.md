---
name: ui-critique
description: Review a UI screenshot or page for clutter, hierarchy, density and polish. Use when asked to critique, declutter or polish a screen (the Dream-RSI watch page, report figures, slides), or when a rendered page looks crowded.
---

# UI critique

Review the rendered screen, not the markup. One verdict, at most five fixes, each with a location.

## When to use

- A screenshot of a page or slide looks crowded, noisy or top-heavy and needs a ranked fix list.
- The watch page (`rsi/watch_static/index.html`) changes and must be re-checked at both viewports.
- A report figure or slide competes with its own labels, chrome or captions.

## Inputs

- Screenshots at 1440x900 and 390x844. Render both; a desktop fix can break the phone.
- The source file, read for the elements behind each defect (class names, `@media` rules).
- Read the image first and name what draws the eye. The markup cannot show density, contrast or
  stacked boxes; only the pixels can.

## Critique pass

Imperative checks. Answer each from the screenshot, then find the element in the source.

**Header**
- Merge status, timestamps and mode tags into one line: app name, status in words, relative timer.
- Delete absolute clock times and mode banners ("demo replay"); keep elapsed time only.
- Keep the header under three text sizes.

**Metrics**
- Collapse the figure grid to one row of badge-like values. Move footnotes to a tooltip or a
  collapsible aside.
- Shrink identity blocks (agent, model, backend) to a single pill.
- Show one unit per value; do not repeat the same number in two places.

**Breadcrumbs**
- Keep the most specific path once, as a small line above the title. Delete any repeat.
- Make the title the largest text in the reader.

**Tabs**
- Use sentence case. Drop uppercase and wide letter-spacing.
- Mark the active tab with a subtle underline or weight change; leave generous gaps.

**Transcript**
- Remove per-turn role labels ("AGENT", "THINKING", tool names) when the avatar and styling
  already carry the role.
- Let text flow like a chat: no box per turn, no oversized left icon column.
- Stream events into the feed; do not refetch and repaint the whole transcript on a timer. Show
  thinking and tool progress as they arrive, and keep the last context visible.
- Under a working agent, an empty transcript must say it is working and will stream ("Agent is
  working. The first streamed event will appear here."). Silence looks like a stall.

**Status and copy**
- State status in words. A decorative live or pulse dot beside the title that repeats the text is
  noise; keep motion only for the one element that is actually running.
- Join hierarchy with ` / ` and adjacent metadata with commas, parentheses or a wider gap. Never
  glue fragments with a middle dot; it is a separator with no meaning.

**Borders**
- Replace nested boxes with one divider or a soft fill.
- Count the visible borders on one screen; each must separate something a reader needs.

**Spacing**
- Give the primary content the widest margins. Padding grows before anything is added.
- Check that labels are small and low-contrast while values are larger: labels never compete.
- A sidebar must fit one viewport before the first datum. Tighten section gaps to 12–16 px before
  accepting a scrollbar.

**Empty states**
- Collapse a zero-data section to one line ("No nodes recorded yet"). Hide its table header, axis
  grid, legend and summary row; an empty chart or a header over "No steps yet" is pure height.
- Keep the section title and a `?` hint; delete the placeholder body.
- Never render em-dash summaries ("Incumbent V —; selected V —") for data that does not exist.

**Disclosures**
- Default reference accordions ("Method and protocol") to collapsed. The reader opens them.
- Use a borderless summary row and remove the browser focus ring on click; keep a subtle
  `:focus-visible` outline for keyboard users. Tabs (`.tab`, `.cyc-tab`) follow the same rule.

**Scrollbars and overflow**
- Style every scrollbar to the palette: transparent track, thin thumb, no arrow steppers. A default
  white OS track on a warm canvas reads as a seam between panes.
- No permanent horizontal track. Reveal the thumb on hover, or replace it with drag-to-pan.
- At 560 px, check the metric ticker and tab row for clipped words. Wrap or scroll them; never cut
  a value mid-letter.

**Jargon**
- Move protocol detail, gate definitions and per-node metadata into accordions or tooltips.
- Keep on-screen text to what a reader needs at a glance.

## Severity

- **Blocker**: the reader cannot find the task title or the live transcript.
- **Major**: a whole region repeats, nests or crowds without adding information.
- **Minor**: a single label, weight or gap is off.

Removing clutter beats adding decoration. A fix that adds a new box, badge or label is suspect;
reject it unless it deletes more than it adds.

## Output format

```
Verdict: <one line, with the worst viewport>
1. [blocker] file:line — <change>. Removes: <what disappears>.
2. [major]   ...
(≤5 fixes, ordered by severity)
Keep: <what works and must survive the edit>
Re-shoot: 1440x900 and 390x844, `python -m http.server` in rsi/watch_static, open index.html?demo=1
```

## Watch-page anti-patterns

Each names the element, not a line number; the file moves.

- **2x2 figure grid**: `.figs` / `.fig` with `.lbl` and `.sub` footnotes. Five cells wrap into a
  2x2 block plus a fifth row on a phone, and the Agent cell (`.fig-model`) spends a whole box on
  one string.
- **Duplicate cycle breadcrumbs**: `#status` (`.status`) says "Cycle 2, exploring" while
  `#rd-kicker` repeats "Node / Cycle 2 / b1-a1".
- **Uppercase tabs**: `.tab` and `.cyc-tab` with `text-transform: uppercase` and wide tracking.
  The five node tabs read as a second window's title bar.
- **Per-turn role labels**: `.who` ("Agent"), `summary.who` ("Thinking"), `.tool .name` next to an
  avatar that already says the same.
- **Bordered bubbles**: `.bubble` and `.tool` carry a border and radius on every turn, so the feed
  stacks as bricks. `.avatar` adds a 24 px circle per row.
- **Header noise**: `.mast` holds a mode label, an absolute clock, a study path, a status line and
  three fact blocks in mixed sizes and cases.
- **Default scrollbars**: `.rail` and `.tx` scroll with the Chromium default track and arrow
  steppers, so a white strip cuts between the rail and the reader.
- **Permanent tree track**: `.tree-wrap` always shows a horizontal scrollbar under the SVG, even
  when the tree fits.
- **Clipped ticker**: `.figs` is a single row with no wrap, so "Policy steps 3 of 6" loses its
  tail at 560 px.
- **Empty ledger**: `table.ledger` keeps its four-column header, a "No steps yet" row and an
  all-dashes summary before the first policy step exists.
- **Empty chart**: `#scores` draws a full 0–50 grid and a three-line caption with zero points.
- **Open reference**: `.about` ("Method and protocol") starts expanded and pushes the rail past one
  viewport; its summary also keeps the default focus ring after a click.
- **Decorative live dot**: `#live-dot` or a pulsing `.pulse` beside the title while `#status`
  already says "Exploring". Redundant motion competes with the transcript.
- **Middle-dot glue**: `· ` joining breadcrumb levels or metadata ("Node · Cycle 2 · b1-a1"). Use
  ` / ` for hierarchy and commas or parentheses for metadata.
- **Silent live transcript**: an empty `.tx` under an `agent` or `evaluating` node with no status
  line. The reader cannot tell a stall from a slow start.
- **Polled feed**: a live view that refetches the whole transcript on a timer, hiding thinking and
  tool progress. Stream events instead and keep the last context visible.

## Model floor

Tested on this page at 390x844 with the same screenshot and prompt:

- DeepSeek V4.1 Flash (low) and GLM-5.3-Flash found the figure grid, cited accurate lines and
  quantified what each fix removes.
- Mistral Small 4 followed the format but missed the figure grid.
- Ministral 3B gave two fixes, pointed at JavaScript lines for CSS defects, and contradicted itself
  in `Keep`.

Do not use a model below GLM-5.3-Flash for this skill.

## Verification

- Re-render 1440x900, 560x900 and 390x844 and compare against the before screenshots.
- Inspect a scrolled pane: the scrollbar must use the palette and show no arrow steppers.
- Confirm the title and transcript dominate at both widths and no text clips or overflows.
- Confirm the page still loads no remote assets (no `http(s)://`, no external `src` or `href`).
- Run `uv run --no-sync pytest -q tests/test_rsi_watch_static.py`.
- Report the viewports shot and the test outcome; state any viewport not checked.
