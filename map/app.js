"use strict";

const DATA_PATHS = {
  values: "../stage-1-pump-it-up/data/TrainingSetValues.csv",
  labels: "../stage-1-pump-it-up/data/TrainingSetLabels.csv",
};

const STATUS_GROUPS = {
  functional: { displayName: "Functional", colour: "#43e7b0" },
  "functional needs repair": { displayName: "Needs repair", colour: "#ffbc55" },
  "non functional": { displayName: "Non-functional", colour: "#ff6175" },
};

// These deliberately broad bounds cover Tanzania and reject the dataset's 1,812
// missing-coordinate placeholders at (0, approximately 0).
const TANZANIA_BOUNDS = {
  west: 28,
  east: 42,
  south: -13,
  north: 0,
};

const POINT_SOURCE_ID = "training-water-points";
const STATUS_LAYER_IDS = {
  functional: { glow: "functional-glow", points: "functional-points" },
  "functional needs repair": { glow: "repair-glow", points: "repair-points" },
  "non functional": { glow: "non-functional-glow", points: "non-functional-points" },
};

const MAP_DETAIL_PRESETS = [
  {
    key: "outline",
    label: "Outline",
    rasterVisibility: "none",
    backgroundColour: "#fbfaf6",
    rasterSaturation: -1,
    rasterContrast: 0,
    rasterBrightnessMin: 0,
    rasterBrightnessMax: 1,
    boundaryFillOpacity: 0.98,
    boundaryLineColour: "#31564b",
    boundaryLineOpacity: 0.78,
    boundaryLineWidth: 1.5,
  },
  {
    key: "quiet",
    label: "Quiet",
    rasterVisibility: "visible",
    backgroundColour: "#d7d9d5",
    rasterSaturation: -0.92,
    rasterContrast: 0.22,
    rasterBrightnessMin: 0.08,
    rasterBrightnessMax: 0.58,
    boundaryFillOpacity: 0,
    boundaryLineColour: "#173d34",
    boundaryLineOpacity: 0.3,
    boundaryLineWidth: 1.1,
  },
  {
    key: "detail",
    label: "Detail",
    rasterVisibility: "visible",
    backgroundColour: "#d9ddd8",
    rasterSaturation: -0.2,
    rasterContrast: 0.06,
    rasterBrightnessMin: 0.04,
    rasterBrightnessMax: 0.88,
    boundaryFillOpacity: 0,
    boundaryLineColour: "#173d34",
    boundaryLineOpacity: 0.2,
    boundaryLineWidth: 0.8,
  },
];

const numberFormatter = new Intl.NumberFormat("en-GB");
const activeStatuses = new Set(Object.keys(STATUS_GROUPS));

let map;
let currentDataset;
let interactionsWired = false;
let hoveredPointId = null;
let mapDetailIndex = 1;

document.addEventListener("DOMContentLoaded", initialisePage);

async function initialisePage() {
  wireControls();
  map = createMap();

  try {
    const mapReady = waitForMapToLoad(map);
    currentDataset = await loadDataset(DATA_PATHS.values, DATA_PATHS.labels);
    await mapReady;
    applyMapDetail();
    plotDataset(currentDataset);
    finishLoading();
  } catch (error) {
    console.error(error);
    showFileFallback(error);
  }
}

function createMap() {
  const createdMap = new maplibregl.Map({
    container: "map",
    center: [35.2, -6.2],
    zoom: 5.15,
    minZoom: 3.5,
    maxZoom: 16,
    attributionControl: false,
    style: {
      version: 8,
      sources: {
        "tanzania-boundary": {
          type: "geojson",
          data: TANZANIA_BOUNDARY,
        },
        osm: {
          type: "raster",
          tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
          tileSize: 256,
          maxzoom: 19,
          attribution: "© OpenStreetMap contributors",
        },
      },
      layers: [
        {
          id: "map-background",
          type: "background",
          paint: { "background-color": "#d7d9d5" },
        },
        {
          id: "osm-tiles",
          type: "raster",
          source: "osm",
          paint: {
            "raster-saturation": -0.92,
            "raster-contrast": 0.22,
            "raster-brightness-min": 0.08,
            "raster-brightness-max": 0.58,
          },
        },
        {
          id: "tanzania-fill",
          type: "fill",
          source: "tanzania-boundary",
          paint: {
            "fill-color": "#fbfaf6",
            "fill-opacity": 0,
          },
        },
        {
          id: "tanzania-border",
          type: "line",
          source: "tanzania-boundary",
          layout: {
            "line-cap": "round",
            "line-join": "round",
          },
          paint: {
            "line-color": "#173d34",
            "line-opacity": 0.3,
            "line-width": 1.1,
          },
        },
      ],
    },
  });

  createdMap.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  createdMap.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");

  return createdMap;
}

function waitForMapToLoad(createdMap) {
  return new Promise((resolve, reject) => {
    createdMap.once("load", resolve);
    createdMap.once("error", (event) => {
      if (!createdMap.loaded()) {
        reject(event.error ?? new Error("The map could not be initialised."));
      }
    });
  });
}

async function loadDataset(valuesSource, labelsSource) {
  setLoadingText("Reading training labels…", "Matching each id to its status group.");

  const statusById = new Map();
  await parseCsvRows(labelsSource, (row) => {
    const id = normaliseId(row.id);
    const status = String(row.status_group ?? "").trim();

    if (id && STATUS_GROUPS[status]) {
      statusById.set(id, status);
    }
  });

  if (statusById.size === 0) {
    throw new Error("No recognised status_group labels were found in the labels CSV.");
  }

  setLoadingText(
    "Locating water points…",
    "Keeping longitude, latitude, status and region; discarding every other field.",
  );

  const features = [];
  const regionCounts = new Map();
  let excludedCoordinates = 0;
  let missingLabels = 0;

  await parseCsvRows(valuesSource, (row) => {
    const status = statusById.get(normaliseId(row.id));
    if (!status) {
      missingLabels += 1;
      return;
    }

    const longitude = Number(row.longitude);
    const latitude = Number(row.latitude);
    const region = String(row.region ?? "").trim();
    if (!isUsableCoordinate(longitude, latitude)) {
      excludedCoordinates += 1;
      return;
    }

    regionCounts.set(region, (regionCounts.get(region) ?? 0) + 1);
    features.push({
      type: "Feature",
      properties: { status, region },
      geometry: {
        type: "Point",
        coordinates: [longitude, latitude],
      },
    });
  });

  if (features.length === 0) {
    throw new Error("No labelled Tanzanian coordinates were found in the values CSV.");
  }

  return {
    geoJson: { type: "FeatureCollection", features },
    regionCounts,
    excludedCoordinates,
    missingLabels,
  };
}

async function parseCsvRows(source, handleRow) {
  const csvText = await readCsvText(source);

  return new Promise((resolve, reject) => {
    const importantErrors = [];

    Papa.parse(csvText, {
      header: true,
      skipEmptyLines: "greedy",
      step(results) {
        for (const error of results.errors ?? []) {
          if (importantErrors.length < 5 && error.code !== "TooFewFields") {
            importantErrors.push(error);
          }
        }

        handleRow(results.data);
      },
      complete() {
        if (importantErrors.length > 0) {
          reject(new Error(`CSV parsing failed: ${importantErrors[0].message}`));
          return;
        }

        resolve();
      },
      error(error) {
        reject(error instanceof Error ? error : new Error(String(error)));
      },
    });
  });
}

async function readCsvText(source) {
  if (typeof source !== "string") {
    return source.text();
  }

  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`Could not load ${source} (${response.status}).`);
  }

  return response.text();
}

function normaliseId(id) {
  return String(id ?? "").trim();
}

function isUsableCoordinate(longitude, latitude) {
  return (
    Number.isFinite(longitude) &&
    Number.isFinite(latitude) &&
    longitude >= TANZANIA_BOUNDS.west &&
    longitude <= TANZANIA_BOUNDS.east &&
    latitude >= TANZANIA_BOUNDS.south &&
    latitude <= TANZANIA_BOUNDS.north
  );
}

function plotDataset(dataset) {
  const existingSource = map.getSource(POINT_SOURCE_ID);
  if (existingSource) {
    existingSource.setData(dataset.geoJson);
  } else {
    map.addSource(POINT_SOURCE_ID, {
      type: "geojson",
      data: dataset.geoJson,
      generateId: true,
    });

    addPointLayers();
    wireMapInteractions();
  }

  updateSummary(dataset);
  applyRegionFilter();
  applyStatusFilter();
  fitMapToData();
}

function addPointLayers() {
  for (const [status, layerIds] of Object.entries(STATUS_LAYER_IDS)) {
    const filter = ["==", ["get", "status"], status];
    const colour = STATUS_GROUPS[status].colour;

    map.addLayer({
      id: layerIds.glow,
      type: "circle",
      source: POINT_SOURCE_ID,
      filter,
      paint: {
        "circle-color": colour,
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 3, 9, 6, 14, 11],
        "circle-opacity": ["interpolate", ["linear"], ["zoom"], 4, 0.12, 10, 0.19],
        "circle-blur": 1,
      },
    });

    map.addLayer({
      id: layerIds.points,
      type: "circle",
      source: POINT_SOURCE_ID,
      filter,
      paint: {
        "circle-color": colour,
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["zoom"],
          4,
          ["case", ["boolean", ["feature-state", "hover"], false], 4, 1.35],
          9,
          ["case", ["boolean", ["feature-state", "hover"], false], 6, 2.8],
          14,
          ["case", ["boolean", ["feature-state", "hover"], false], 9, 5.5],
        ],
        "circle-opacity": ["interpolate", ["linear"], ["zoom"], 4, 0.68, 9, 0.88],
        "circle-stroke-color": "rgba(5, 15, 16, 0.72)",
        "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 4, 0.25, 12, 1],
      },
    });
  }
}

function wireMapInteractions() {
  if (interactionsWired) {
    return;
  }

  interactionsWired = true;
  const popup = new maplibregl.Popup({
    closeButton: false,
    closeOnClick: false,
    offset: 10,
  });

  const showHoveredPoint = (event) => {
    const feature = event.features?.[0];
    if (!feature) {
      return;
    }

    if (hoveredPointId !== null) {
      map.setFeatureState({ source: POINT_SOURCE_ID, id: hoveredPointId }, { hover: false });
    }

    hoveredPointId = feature.id;
    map.setFeatureState({ source: POINT_SOURCE_ID, id: hoveredPointId }, { hover: true });
    map.getCanvas().style.cursor = "pointer";
    showPointPopup(popup, feature);
  };

  const clearHoveredPoint = () => {
    if (hoveredPointId !== null) {
      map.setFeatureState({ source: POINT_SOURCE_ID, id: hoveredPointId }, { hover: false });
    }

    hoveredPointId = null;
    map.getCanvas().style.cursor = "";
    popup.remove();
  };

  const showClickedPoint = (event) => {
    const feature = event.features?.[0];
    if (feature) {
      showPointPopup(popup, feature);
    }
  };

  for (const { points: pointLayerId } of Object.values(STATUS_LAYER_IDS)) {
    map.on("mousemove", pointLayerId, showHoveredPoint);
    map.on("mouseleave", pointLayerId, clearHoveredPoint);
    map.on("click", pointLayerId, showClickedPoint);
  }
}

function showPointPopup(popup, feature) {
  const [longitude, latitude] = feature.geometry.coordinates;
  const status = feature.properties.status;
  const statusDetails = STATUS_GROUPS[status];

  popup
    .setLngLat([longitude, latitude])
    .setHTML(`
      <div class="well-popup-status" style="--popup-colour: ${statusDetails.colour}">
        <span class="well-popup-dot"></span>
        ${statusDetails.displayName}
      </div>
      <div class="well-popup-coordinates">${latitude.toFixed(5)}, ${longitude.toFixed(5)}</div>
    `)
    .addTo(map);
}

function wireControls() {
  document.querySelector("#map-detail-toggle").addEventListener("click", () => {
    mapDetailIndex = (mapDetailIndex + 1) % MAP_DETAIL_PRESETS.length;
    applyMapDetail();
  });

  document.querySelector("#region-filter").addEventListener("change", () => {
    const features = applyRegionFilter();
    fitMapToFeatures(features);
  });

  for (const button of document.querySelectorAll(".legend-item")) {
    button.addEventListener("click", () => {
      const status = button.dataset.status;
      if (activeStatuses.has(status)) {
        activeStatuses.delete(status);
      } else {
        activeStatuses.add(status);
      }

      applyStatusFilter();
    });
  }

  document.querySelector("#file-fallback").addEventListener("submit", async (event) => {
    event.preventDefault();
    const valuesFile = document.querySelector("#values-file").files[0];
    const labelsFile = document.querySelector("#labels-file").files[0];

    if (!valuesFile || !labelsFile) {
      return;
    }

    resetLoadingState();
    try {
      currentDataset = await loadDataset(valuesFile, labelsFile);
      if (!map.loaded()) {
        await waitForMapToLoad(map);
      }
      plotDataset(currentDataset);
      finishLoading();
    } catch (error) {
      console.error(error);
      showFileFallback(error);
    }
  });
}

function applyMapDetail() {
  const preset = MAP_DETAIL_PRESETS[mapDetailIndex];
  const toggle = document.querySelector("#map-detail-toggle");
  document.querySelector("#map-detail-label").textContent = preset.label;
  document.querySelector(".map-shell").dataset.mapDetail = preset.key;
  toggle.title = `Background detail: ${preset.label}. Click for the next level.`;
  toggle.setAttribute("aria-label", toggle.title);

  if (!map || !map.isStyleLoaded()) {
    return;
  }

  map.setLayoutProperty("osm-tiles", "visibility", preset.rasterVisibility);
  map.setPaintProperty("map-background", "background-color", preset.backgroundColour);
  map.setPaintProperty("osm-tiles", "raster-saturation", preset.rasterSaturation);
  map.setPaintProperty("osm-tiles", "raster-contrast", preset.rasterContrast);
  map.setPaintProperty("osm-tiles", "raster-brightness-min", preset.rasterBrightnessMin);
  map.setPaintProperty("osm-tiles", "raster-brightness-max", preset.rasterBrightnessMax);
  map.setPaintProperty("tanzania-fill", "fill-opacity", preset.boundaryFillOpacity);
  map.setPaintProperty("tanzania-border", "line-color", preset.boundaryLineColour);
  map.setPaintProperty("tanzania-border", "line-opacity", preset.boundaryLineOpacity);
  map.setPaintProperty("tanzania-border", "line-width", preset.boundaryLineWidth);
}

function applyStatusFilter() {
  const totalStatusCount = Object.keys(STATUS_GROUPS).length;
  const filterSummary = document.querySelector("#filter-summary");
  filterSummary.textContent = `${activeStatuses.size}/${totalStatusCount}`;
  filterSummary.title = `${activeStatuses.size} of ${totalStatusCount} status groups shown`;

  for (const button of document.querySelectorAll(".legend-item")) {
    const status = button.dataset.status;
    const isShown = activeStatuses.has(status);
    button.setAttribute("aria-pressed", String(isShown));
    button.title = `${STATUS_GROUPS[status].displayName}: click to ${isShown ? "hide" : "show"}`;
  }

  if (currentDataset) {
    updateCountsForFeatures(getSelectedRegionFeatures());
  }

  if (!map) {
    return;
  }

  for (const [status, layerIds] of Object.entries(STATUS_LAYER_IDS)) {
    const visibility = activeStatuses.has(status) ? "visible" : "none";
    for (const layerId of Object.values(layerIds)) {
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, "visibility", visibility);
      }
    }
  }
}

function updateSummary(dataset) {
  document.querySelector("#excluded-count").textContent = numberFormatter.format(
    dataset.excludedCoordinates,
  );

  populateRegionOptions(dataset);
}

function populateRegionOptions(dataset) {
  const select = document.querySelector("#region-filter");
  const previousValue = select.value;
  select.replaceChildren();

  const allRegionsOption = document.createElement("option");
  allRegionsOption.value = "";
  allRegionsOption.textContent = `All regions (${numberFormatter.format(dataset.geoJson.features.length)})`;
  select.append(allRegionsOption);

  const regionNames = [...dataset.regionCounts.keys()].filter(Boolean).sort((left, right) =>
    left.localeCompare(right, "en-GB"),
  );

  for (const region of regionNames) {
    const option = document.createElement("option");
    option.value = region;
    option.textContent = `${region} (${numberFormatter.format(dataset.regionCounts.get(region))})`;
    select.append(option);
  }

  select.value = regionNames.includes(previousValue) ? previousValue : "";
  select.disabled = false;
}

function applyRegionFilter() {
  const features = getSelectedRegionFeatures();
  const source = map?.getSource(POINT_SOURCE_ID);
  if (source) {
    hoveredPointId = null;
    source.setData({ type: "FeatureCollection", features });
  }

  updateCountsForFeatures(features);

  const select = document.querySelector("#region-filter");
  const selectedName = select.value || "All regions";
  select.title = `${selectedName}: ${numberFormatter.format(features.length)} mapped wells`;
  return features;
}

function getSelectedRegionFeatures() {
  if (!currentDataset) {
    return [];
  }

  const selectedRegion = document.querySelector("#region-filter").value;
  if (!selectedRegion) {
    return currentDataset.geoJson.features;
  }

  return currentDataset.geoJson.features.filter(
    (feature) => feature.properties.region === selectedRegion,
  );
}

function updateCountsForFeatures(features) {
  const counts = Object.fromEntries(Object.keys(STATUS_GROUPS).map((status) => [status, 0]));
  for (const feature of features) {
    counts[feature.properties.status] += 1;
  }

  for (const [status, count] of Object.entries(counts)) {
    document.querySelector(`[data-count-for="${status}"]`).textContent = numberFormatter.format(count);
  }

  const visibleCount = [...activeStatuses].reduce(
    (total, status) => total + counts[status],
    0,
  );
  document.querySelector("#visible-count").textContent = numberFormatter.format(visibleCount);
}

function fitMapToData() {
  if (!currentDataset) {
    return;
  }

  fitMapToFeatures(currentDataset.geoJson.features);
}

function fitMapToFeatures(features) {
  if (!map || features.length === 0) {
    return;
  }

  const bounds = new maplibregl.LngLatBounds();
  for (const feature of features) {
    bounds.extend(feature.geometry.coordinates);
  }

  const compactLayout = window.matchMedia("(max-width: 1100px)").matches;
  map.fitBounds(bounds, {
    padding: compactLayout
      ? { top: 130, right: 28, bottom: 28, left: 28 }
      : { top: 90, right: 70, bottom: 55, left: 70 },
    maxZoom: 8,
    duration: 1200,
  });
}

function setLoadingText(title, message) {
  document.querySelector("#load-title").textContent = title;
  document.querySelector("#load-message").textContent = message;
}

function finishLoading() {
  setLoadingText("Map ready", `${numberFormatter.format(currentDataset.geoJson.features.length)} locations plotted.`);
  window.setTimeout(() => {
    document.querySelector("#load-overlay").classList.add("is-complete");
  }, 300);
}

function showFileFallback(error) {
  const overlay = document.querySelector("#load-overlay");
  overlay.classList.remove("is-complete");
  overlay.classList.add("is-error");
  document.querySelector("#load-title").textContent = "Choose the local training files";
  document.querySelector("#load-message").textContent =
    error?.message || "The expected training CSVs could not be loaded.";
  document.querySelector("#file-fallback").hidden = false;
}

function resetLoadingState() {
  const overlay = document.querySelector("#load-overlay");
  overlay.classList.remove("is-error", "is-complete");
  document.querySelector("#file-fallback").hidden = true;
  setLoadingText("Reading selected files…", "Joining the training values and labels locally.");
}
