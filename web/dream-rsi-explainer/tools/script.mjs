// Parses script.txt into scenes -> sentences -> tokens (shown text, spoken text, emphasis, cue).
export function parseScript(text) {
  const scenes = [];
  let scene = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || (line.startsWith("#") && !/^#\s*\w+\s*\|/.test(line))) continue;
    const head = line.match(/^#\s*(\w+)\s*\|\s*(.+)$/);
    if (head) {
      scene = { id: head[1], chip: head[2].trim(), sentences: [] };
      scenes.push(scene);
      continue;
    }
    const tokens = [];
    let emph = false;
    for (const rawTok of line.match(/(?:\{[^}]*\}|\S)+/g)) {
      const m = rawTok.match(/^(?:@(\w+):)?(\*)?(\{[^}]*\}|[^*]+?)(\*)?([.,!?;:]*)$/);
      if (!m) throw new Error(`bad token: ${rawTok}`);
      const [, cue, open, core, close, punct] = m;
      if (open) emph = true;
      let shown = core;
      let spoken = core;
      if (core.startsWith("{")) [shown, spoken] = core.slice(1, -1).split("|");
      tokens.push({ shown, spoken, emph, cue: cue || null, punct });
      if (close) emph = false;
    }
    const spokenText = tokens.map((t) => t.spoken + t.punct).join(" ");
    scene.sentences.push({ tokens, spokenText });
  }
  return scenes;
}
