"use strict";

(function loadProjectStatus() {
  const statusUrl = document.documentElement.dataset.statusUrl;
  if (!statusUrl) {
    return;
  }

  const integerFormatter = new Intl.NumberFormat("en-GB");

  fetch(statusUrl, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Project status request failed with ${response.status}.`);
      }

      return response.json();
    })
    .then((status) => {
      bindText(status);
      bindFractions(status);
      bindProgress(status);
      document.documentElement.dataset.statusLoaded = "true";
      document.dispatchEvent(new CustomEvent("projectstatusloaded", { detail: status }));
    })
    .catch((error) => {
      console.warn("Using the embedded project-status snapshot.", error);
    });

  function bindText(status) {
    for (const element of document.querySelectorAll("[data-status-text]")) {
      const value = readPath(status, element.dataset.statusText);
      if (value !== undefined && value !== null) {
        element.textContent = formatValue(value, element.dataset.statusFormat);
      }
    }
  }

  function bindFractions(status) {
    for (const element of document.querySelectorAll("[data-status-fraction-current]")) {
      const current = readPath(status, element.dataset.statusFractionCurrent);
      const total = readPath(status, element.dataset.statusFractionTotal);
      if (Number.isFinite(current) && Number.isFinite(total)) {
        element.textContent = `${integerFormatter.format(current)} / ${integerFormatter.format(total)}`;
      }
    }
  }

  function bindProgress(status) {
    for (const element of document.querySelectorAll("[data-status-progress-current]")) {
      const current = readPath(status, element.dataset.statusProgressCurrent);
      const total = readPath(status, element.dataset.statusProgressTotal);
      if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
        continue;
      }

      const percentage = Math.max(0, Math.min(100, (current / total) * 100));
      element.style.setProperty("--status-progress", `${percentage}%`);
      element.setAttribute("aria-label", `${Math.round(percentage)} per cent complete`);

      if (element instanceof HTMLProgressElement) {
        element.value = current;
        element.max = total;
      }
    }
  }

  function readPath(source, path) {
    return path.split(".").reduce((value, segment) => value?.[segment], source);
  }

  function formatValue(value, format) {
    switch (format) {
      case "integer":
        return integerFormatter.format(value);
      case "score":
        return Number(value).toFixed(4);
      case "percentage-from-score":
        return `${(Number(value) * 100).toFixed(2)}%`;
      default:
        return String(value);
    }
  }
})();
