/* Theme bootstrap: runs before first paint so a dark preference never flashes light. */
(function () {
  try {
    var choice = localStorage.getItem("watch.theme") || "system";
    var dark =
      choice === "dark" ||
      (choice === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    document.documentElement.dataset.themeChoice = choice;
  } catch (error) {
    document.documentElement.dataset.theme = "light";
  }
})();
