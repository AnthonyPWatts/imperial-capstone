# Module 12 intake and submission rules

Verified against the course on **24 September 2026**. The initial observations are available locally; modelling and submissions have not started as part of this collection.

## Submission frequency

The limit is **one evaluated query per function per week**: eight function evaluations per round, over **13 rounds, Modules 12–24**. This gives 13 additional observations per function, or 104 across all eight, assuming all rounds are completed. It does not support hourly evaluated queries.

The portal guide says pending inputs can be updated until the week ends or processing occurs. It does not state a numerical limit on such edits. These updates replace the pending inputs; they are not extra observations. The next submission becomes available after the current one is processed.

The FAQ says normal submissions are processed at the end of each module. Its late-submission guidance is **24–48 hours**, while the portal guide says late entries are processed immediately. Treat that as an unresolved documentation inconsistency, not a way to obtain more evaluations. The weekly query allocation and 13-round total are explicit in the lesson and portal guide.

Sources: [Mini-lesson 12.7](https://classroom.emeritus.org/courses/18642/modules/items/3237302), [portal guide, pp. 3–6](https://classroom.emeritus.org/courses/18642/files/6691294?wrap=1), [FAQ, pp. 8–9](https://classroom.emeritus.org/courses/18642/files/6691536?wrap=1). These are published rules; no test submission was made and the authenticated portal controls were not tested.

## First submission and reflection

- The first [Canvas capstone component](https://classroom.emeritus.org/courses/18642/modules/items/3237304) is due **1 October 2026 at 17:29 BST**. Check the portal's Cohorts page for the query deadline rather than assuming that it matches Canvas.
- Propose one point for every function. Every coordinate must be in `[0, 1)`, with exactly six decimal places and no spaces: for example, `0.123456-0.654321`.
- Maximise the supplied outputs for all eight functions. Negative scores can be valid; do not negate them again because the application analogy describes minimisation.
- Submit numeric inputs through the portal, then a reflection of **under 700 words** on the discussion board. Explain the selection principle, the hardest functions and missing information, and planned changes for the next round. The instructions also ask for thoughtful peer comments.
- Record the selected inputs, returned outputs, model configuration and reasoning for each round. Preserve the original arrays and store accumulated observations separately.

## Initial data

The [official ZIP](https://mo-pcco.s3.us-east-1.amazonaws.com/IMP-PCMLAI-R3-Master-EMCODE/module12/Initial_data_points_starter.zip) is retained unchanged under `data/Initial_data_points_starter.zip`. Extracted files are at `data/initial_data/function_N/initial_inputs.npy` and `initial_outputs.npy`.

| Function | Input shape | Output shape | Course analogy / feature |
| --- | --- | --- | --- |
| 1 | `(10, 2)` | `(10,)` | Locating contamination sources; highly local signal |
| 2 | `(10, 2)` | `(10,)` | Noisy log-likelihood with local optima |
| 3 | `(15, 3)` | `(15,)` | Drug combinations; transformed adverse-reaction score |
| 4 | `(30, 4)` | `(30,)` | Warehouse-model tuning with expensive evaluations |
| 5 | `(20, 4)` | `(20,)` | Chemical yield; described as typically unimodal |
| 6 | `(20, 5)` | `(20,)` | Recipe quality with negative contributions |
| 7 | `(30, 6)` | `(30,)` | Six model hyperparameters |
| 8 | `(40, 8)` | `(40,)` | Eight-parameter optimisation |

All 16 arrays loaded with `allow_pickle=False`. Shapes match the [function descriptions](https://classroom.emeritus.org/courses/18642/modules/items/3237303); values are finite, inputs lie in `[0, 1)`, and each function has distinct input rows. There are **175 initial observations**. The ZIP, arrays and local verification manifest are Git-ignored.

The generic “ten starting points” wording in Mini-lesson 12.7 and the assignment is an oversimplification. Use all supplied observations, as clarified by the FAQ and confirmed by the files. The FAQ's example feedback includes coordinates above 1; those examples do not override the explicit input bounds.

## Other capstone milestones

- The first optional 30-minute consultation runs **1–8 October 2026**. Book at least 24 hours ahead. [Consultation instructions](https://classroom.emeritus.org/courses/18642/modules/items/3237313) provide the facilitator links; Section A uses [Murari Ramuka](https://calendly.com/murari_045/capstone-consultation).
- The second consultation is **7–21 January 2027**, with scheduling opening **17 December 2026**, according to the [24 September announcement](https://classroom.emeritus.org/courses/18642/announcements/990157).
- The leaderboard becomes available near the end of Module 24 after submissions are processed.
- Module 25 is the final public GitHub repository submission. Retain reproducible notebooks for all eight functions, progress plots, best observed inputs/outputs, rationale and limitations, a roughly 100-word non-technical summary, a datasheet and a model card. Existing documentation templates cover these areas.

Course PDFs remain in the parent course archive. This public capstone repository retains concise working notes and source links rather than copies of course teaching material or account credentials.
