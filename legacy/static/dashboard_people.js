(function () {
  const composer = document.getElementById("peopleComposer");
  const openBtn = document.getElementById("openPeopleComposer");
  const randomForm = document.getElementById("peopleRandomForm");
  const manualForm = document.getElementById("peopleManualForm");
  const modeButtons = document.querySelectorAll("[data-people-mode]");

  if (!composer || !openBtn || !randomForm || !manualForm) {
    return;
  }

  function setMode(mode) {
    const useManual = mode === "manual";
    manualForm.hidden = !useManual;
    randomForm.hidden = useManual;
    modeButtons.forEach((btn) => {
      btn.classList.toggle("is-active", btn.getAttribute("data-people-mode") === mode);
    });
  }

  openBtn.addEventListener("click", function () {
    composer.hidden = !composer.hidden;
    if (!composer.hidden) {
      setMode("random");
    }
  });

  modeButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
      setMode(btn.getAttribute("data-people-mode") || "random");
    });
  });
})();
