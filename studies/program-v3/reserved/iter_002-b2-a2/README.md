# iter_002-b2-a2, reserved 500

`final/` is the single complete reserved run.

`attempt-1-interrupted/` is the first launch. The tooling that held the process (a background shell
with a time limit) killed it after 123 of 500 episodes; its `result.json` (status `running`) is left out. No
result was read or used before the rerun. The cell was run once more, from the start, detached from
that shell, with the same workspace, protocol and environment. Its partial episodes are kept as
evidence only and are not pooled.
