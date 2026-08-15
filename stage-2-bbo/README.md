# Stage 2: black-box optimisation

This is the assessed capstone problem: maximise eight unknown functions using
only their observed inputs and outputs. The functions range from two to eight
dimensions. Each course round adds a portal observation to the local dataset.

Browse the [Stage 2 workspace](https://anthonypwatts.github.io/imperial-capstone/stage-2-bbo/)
or return to the [Capstone Hub](https://anthonypwatts.github.io/imperial-capstone/).

## Intended approach

1. Load and validate the supplied NumPy arrays for all eight functions.
2. Plot the observations and establish a random-search reference point.
3. Scale the inputs and fit a Gaussian-process surrogate.
4. Compare expected improvement with an upper-confidence bound.
5. Propose one valid point per function, submit it through the portal and log
   the result.
6. Refit after each observation and explain any change in strategy.

This keeps the method aligned with the course material without turning the
exercise into a research project. The useful business analogue is sequential
decision-making where experiments are expensive: marketing tests, process
settings, pricing trials or product configurations.

## Repository structure

| Path | Contents |
| --- | --- |
| `data/` | Locally supplied input and output arrays; ignored by Git |
| `notebooks/` | Exploration, surrogate modelling and acquisition experiments |
| `src/` | Reusable BBO code |
| `results/` | Observation and experiment logs safe to publish |
| `submissions/` | Proposed points and portal-submission notes |
| `docs/` | Datasheet, model card and non-technical final summary |

## Constraints to enforce

- Work on all eight functions.
- Keep every coordinate in the interval `[0, 1)`.
- Format portal submissions to six decimal places as hyphen-separated
  coordinates with no spaces.
- Check dimensions, duplicates and bounds before proposing a point.
- Preserve an auditable link between every submitted point, returned value,
  model configuration and source commit.

Follow the portal and Module 12 instructions if their submission rules differ
from this scaffold.
