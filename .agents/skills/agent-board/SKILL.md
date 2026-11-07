---
name: agent-board
description: Read or post to AGENT_BOARD.jsonl (claims, results, questions, corrections) shared by Fernando's (F) and Noah's (N) agents. Use at task start, when claiming shared work, when a result is ready, and to correct an earlier record.
---

# Agent board

`AGENT_BOARD.md` is the protocol; `AGENT_BOARD.jsonl` is the append-only log.

## Read

```sh
git fetch origin main
git show origin/main:AGENT_BOARD.jsonl | tail -n 10
```

## Post

```sh
python scripts/agent_board.py append --from F --to '*' --type CLM --scope <scope> \
  --msg "One line: what you own and what you will not touch."
python scripts/agent_board.py append --from F --to '*' --type DONE --ref <CLM id> \
  --scope <scope> --msg "One line: result, meaning, caveat, next step."
```

`append` validates the whole log first. A pre-existing malformed record makes it refuse; report
that instead of editing old lines.

Publishing a board commit follows the user's authorization for that session. When authorized:

```sh
git add AGENT_BOARD.jsonl && git commit -m "board: <summary>" && git push origin main
```

If the push is rejected, `git pull --rebase origin main`, check that no ID is duplicated, push
again. Never force-push the board.

## Rules

- `--from` is your own participant. Never post as the other agent.
- Never edit or delete an existing line. Correct with `--type CORR --ref <id>`.
- `ACK`, `DONE`, `BLK` and `CORR` need `--ref`.
- Messages follow `terse-writing`; results must pass `reporting-results`.
