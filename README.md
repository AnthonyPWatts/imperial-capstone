# Imperial ML and AI capstone

Coursework and portfolio work for the Imperial College London Professional
Certificate in Machine Learning and Artificial Intelligence.

## Project structure

| Area | Purpose |
| --- | --- |
| [`index.html`](index.html) | Local Capstone Hub joining the project stages and working tools |
| [`stage-1-pump-it-up/`](stage-1-pump-it-up/) | Applied multiclass classification using the DrivenData *Pump It Up* competition |
| [`stage-2-bbo/`](stage-2-bbo/) | The assessed black-box optimisation (BBO) capstone covering eight unknown functions |
| [`dashboard/`](dashboard/) | Static plan and progress dashboard for the Stage 1 competition |
| [`map/`](map/) | Local-only interactive map of the Stage 1 training labels |

Stage 1 is a self-contained machine-learning project used to practise the
course workflow on a real operational problem. Stage 2 follows the course's
fixed BBO brief and will be developed as each set of observations is released.

## Local project hub

Run the static site from this directory so that the hub, shared status snapshot
and local-data map all resolve from the same origin:

```powershell
python -m http.server 8000
```

Then open <http://localhost:8000/>. The two stage cards lead into separate
workspaces; stage-specific tools and progress are available only after selecting
a stage. Stage 2 remains an explicit empty state until its course inputs and
dates are available.
Non-sensitive stage labels and dashboard headline values live in
[`project-status.json`](project-status.json). Detailed evidence and modelling
decisions remain in the relevant notebooks, reports and maintainer notes.

## Working conventions

- Keep exploratory work in numbered notebooks.
- Move code worth reusing into `src/`.
- Record decisions and results rather than relying on notebook output alone.
- Do not commit competition downloads, supplied BBO observations or secrets.
- Prefer a simple, reproducible baseline before tuning more complex models.

The scaffold gives each part of the modelling work a home. Each stage README
describes the immediate plan and the evidence still missing.
