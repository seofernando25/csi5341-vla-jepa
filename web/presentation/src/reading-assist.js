/* Visual emphasis only. Original text, punctuation and spaces remain intact. */
window.ReadingAssist = (() => {
  const key = "notes-bold-prefix";
  const enabled = () => localStorage.getItem(key) !== "false";
  function render(target, text, on = enabled()) {
    const fragment = document.createDocumentFragment();
    if (!on) fragment.append(document.createTextNode(text));
    else {
      let offset = 0;
      for (const match of text.matchAll(
        /\p{L}[\p{L}\p{M}]*(?:['’][\p{L}\p{M}]+)*/gu,
      )) {
        fragment.append(
          document.createTextNode(text.slice(offset, match.index)),
        );
        const word = Array.from(match[0]);
        const length = Math.min(3, Math.ceil(word.length / 2));
        const prefix = document.createElement("b");
        prefix.className = "reading-prefix";
        prefix.textContent = word.slice(0, length).join("");
        fragment.append(
          prefix,
          document.createTextNode(word.slice(length).join("")),
        );
        offset = match.index + match[0].length;
      }
      fragment.append(document.createTextNode(text.slice(offset)));
    }
    target.replaceChildren(fragment);
  }
  function bind(button, rerender) {
    function label() {
      button.textContent = "Bold word starts";
      button.setAttribute("aria-pressed", String(enabled()));
    }
    button.onclick = () => {
      localStorage.setItem(key, String(!enabled()));
      label();
      rerender();
    };
    label();
  }
  return { enabled, render, bind };
})();
