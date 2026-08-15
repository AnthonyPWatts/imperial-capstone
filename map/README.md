# Training label map

This single-purpose page plots the Pump It Up training coordinates and colours
each point by `status_group`. Visitors can filter the points by region.

Open the [live training label map](https://anthonypwatts.github.io/imperial-capstone/map/)
from the [Capstone Hub](https://anthonypwatts.github.io/imperial-capstone/).
GitHub Pages does not contain the competition CSVs, so select local copies with
the page's file pickers.

## Run it locally

From the repository root:

```powershell
python -m http.server 8000
```

Then open <http://localhost:8000/map/>.

Open the local Capstone Hub at <http://localhost:8000/> as the shared entry
point for the map, status dashboard and project stages.

The page expects these ignored local files:

- `stage-1-pump-it-up/data/TrainingSetValues.csv`
- `stage-1-pump-it-up/data/TrainingSetLabels.csv`

If they are not available at those paths, the page offers local file pickers. This also makes it possible to open `index.html` directly.

## What the browser retains

The two raw files are joined by `id` in the browser. The map data then contains only:

- longitude;
- latitude;
- `status_group`;
- region.

No derived competition data is written to the repository or sent to a backend. The browser does request OpenStreetMap basemap tiles for the visible area.

The compact toolbar cycles through outline, quiet and detailed backgrounds. The outline uses an embedded, simplified [Natural Earth](https://github.com/nvkelso/natural-earth-vector) Tanzania boundary so it remains available when the page is opened directly from disk.

Locations outside broad Tanzanian coordinate bounds are excluded. In the current training set this removes 1,812 placeholder coordinates at `(0, approximately 0)` and leaves 57,588 plotted locations.

## Libraries

- [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/) renders the map and point layers using WebGL.
- [Papa Parse](https://www.papaparse.com/docs) handles CSV quoting and row parsing; the page retains only the joined map fields after parsing.

Both libraries are pinned and loaded from unpkg; there is no build step.
