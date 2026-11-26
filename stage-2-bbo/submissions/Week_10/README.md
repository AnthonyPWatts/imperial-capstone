# Week 10 query decision

**Prepared and unsubmitted on 25 November 2026.** Round 9 is the latest
confirmed feedback. Four opportunities remain per function, including Week 10.

- [Revised cohort-informed batch](submissions.txt) uses the historical
  challengers and deliberately repeats a published F2 input. Use this batch
  if the course permits cohort-informed assessed selection and published-input
  reuse.
- [Official-data fallback](official-data-submissions.txt) is the batch to use
  if the assessed-use rules require official data only or remain unclear close
  to the portal deadline.
- [Initial proposal](initial-proposal.md) and the [80/10/10 advisory experiment](advisory-review.md)
  preserve the earlier decisions separately from this shortlist review.

## Revised cohort-informed batch

| Function | Input | Rationale |
| --- | --- | --- |
| 1 | `0.628558-0.628558` | Maximise the seven-positive official source fit along the diagonal supported by the archived peak. The local archive fit predicts about 1.999997 at this previously untested input. |
| 2 | `0.500000-0.500000` | Deliberate external replication. Eight archived returns at this exact point scored 0.756–0.920, all above the official best of 0.723. The official GP favours the shortlist winner at x1 ≈ 0.70, with mean 0.645 versus 0.175 here, so this is an explicit transfer test. |
| 3 | `0.520000-0.555000-0.450000` | Keep x1 and x3 fixed and test 0.005 from Aby's x2 setting, which returned −0.000140. The ensemble winner at x2 ≈ 0.95 relies on a positive extrapolation beyond every measured F3 score in the reviewed sets. |
| 4 | `0.419096-0.406060-0.386528-0.426466` | Test the x3 midpoint of the two best official returns, holding x1, x2 and x4 fixed. The official Matérn mean is 0.688 versus the 0.666 incumbent. |
| 5 | `0.999999-0.999999-0.999999-0.959224` | Isolate x4 while meeting the 0.020 spacing rule. Expect a lower return than the corner. The query measures the response farther down that arm. |
| 6 | `0.415446-0.370229-0.652604-0.763869-0.109806` | Test the centroid of four distinct archive inputs in the best interior cluster. They returned −0.065 to −0.006 and came from one participant. The official GP favours the boundary winner, −0.200 versus −0.290 here. |
| 7 | `0.216539-0.157590-0.452435-0.276118-0.307725-0.659013` | Use the coordinatewise median of six distinct archived inputs above 3.3 and one corroborating independent input. The official GP favours the boundary winner, 2.735 versus 2.289 here. The move targets the region supported by both x1 and x3. |
| 8 | `0.100000-0.150000-0.130000-0.150000-0.800000-0.500000-0.200000-0.600000` | Use the starter-only quadratic's fitted peak at 10.0. It reproduced all nine subsequent official returns within 2×10⁻¹⁴. The same input appears in an independent campaign. |

The F2, F6 and F7 challengers are explicit departures from the bounded
advisory shortlist. The official round supplies an independent transfer check.
[Advisory comparison](advisory-review.md) · [Archive evidence](../../side-quests/historical-cohort-2024/README.md)

## Official-data fallback

F1 uses the unconstrained seven-positive official source peak. F2 uses the
leading official two-query candidate near its strongest measured band. F3
stays on the official high-score line and moves x2 from 0.61 to 0.65. F6 and
F7 use local official Matérn shortlist points on their best observed faces.
F4, F5 and F8 use the same inputs as the revised batch. All eight coordinates
in this fallback can be derived from the official observations. The F8 model
form was checked after the external review and its coefficients and peak
are reproducible from the official starters alone.
[Official weekly reviews](../../results/week-10-review.md#weekly-decision-notebooks)

## Course and portal check

- The archived Module 19 Canvas brief lists **26 November 2026 at 16:29 GMT**
  for the module activities. Required discussion 19.1 is a transformer
  explainer. Capstone component 19.1 is a separate strategy reflection
  under 700 words. Its introduction refers to batch 8 and its prompts refer
  to batch 7. The archived wording supplies no transformer requirement for
  the numerical portal queries. [Transformer discussion](https://classroom.emeritus.org/courses/18642/discussion_topics/969740?module_item_id=3237593) · [Capstone reflection](https://classroom.emeritus.org/courses/18642/discussion_topics/969741?module_item_id=3237612)
- The **25 November Week 9 feedback** directs learners to the deadline shown
  in the Capstone Portal. Use that deadline for the numeric batch. The exact
  Week 10 portal deadline and the next authenticated assignment instructions
  were unverified on 25 November. The portal guide permits updates to a
  pending batch until the week ends or it is processed. An update replaces
  the pending inputs. [Portal rules and sources](../../docs/module-12-intake.md#submission-frequency)
- Imperial's public syllabus lists **Advanced Generative AI and LLMs**
  for Module 20 and **Transparency and Interpretability** for Module 21.
  These titles establish the teaching topics. Use the saved owner/adviser
  comparison and pairwise uncertainty slices as evidence if the new brief
  asks for explanation. The frozen transformer comparison is available
  as a modelling experiment if the brief asks for it.
  [Imperial syllabus](https://www.imperial.ac.uk/business-school/executive-education/technology-analytics-data-science/professional-certificate-machine-learning-and-artificial-intelligence-programme/online/) · [Transformer benchmark](../../results/week-10-review.md#pooled-validation-and-tabular-transformer-experiment)
- Check the new Canvas activity and the portal's Cohorts page before using
  either batch. Public availability and the shared starters establish
  provenance and comparability. The course sources reviewed leave permission
  for assessed use of peer or archive scores unresolved.
  [Course data-use review](../../docs/capstone-data-policy.md)

Use the Week 10 returns to decide the Week 11 direction. A new best at the F2
centre or in the F6/F7 interiors supports local refinement there. A weak
return supports going back to the established official region while
retaining the transfer result in the decision trail.

## Verification

Both files contain eight lines in function order. The inputs were checked
against the portal's dimensions, bounds and six-decimal format, the nine
previous submissions, the current observations and the per-function
spacing rule. The archive already records the revised F2 coordinate.
An independent campaign records the F8 coordinate.
