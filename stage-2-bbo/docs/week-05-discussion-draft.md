# Week 5 discussion draft

**Status: posted to the course discussion board.** The project owner confirmed
posting on **29 October 2026**; the exact posting timestamp was not supplied.
This file preserves the final draft agreed before that confirmation.

Course activity: [Stage 2 capstone component 15.1](https://classroom.emeritus.org/courses/18642/discussion_topics/969759).
The seven paragraphs below follow the seven reflection prompts in order.

## Discussion text

When deciding which inputs might act like support vectors and guide queries, F3 made me cautious. Its RBF SVM used 14 of 19 observations, including (0.954524, 0.819470, 0.042163). That point’s fitted decision score was +0.054, close to the model’s zero boundary. Support vectors can include margin violations, so their number alone doesn’t validate the boundary. Leave-one-out predictions labelled every point “bad”, missing all five good results: 50% balanced accuracy. I’d use the better-performing regression surrogate for query selection and retain exploration, rather than target this classifier’s boundary.

I examined how surrogate predictions changed with the inputs, calculating neural gradients through backpropagation and checking them against finite differences. This checked the derivative calculation, not its agreement with the hidden function. F5’s upper corner, (0.999999, 0.999999, 0.999999, 0.999999), returned 8662.405001; subsequent nearby queries returned 8525.75 and 8449.98. All four neural derivatives were positive, suggesting small decreases in the coordinates would lower the predicted score. We are maximising, but the bounded ascent trial could take no step. I suspect we may already have found the maximum, or be very close, although this does not establish a global maximum.

To frame BBO as classification, I labelled outcomes reaching the training data’s 75th percentile “good”, recalculating the threshold within each validation fold. Logistic regression fits a straight boundary; RBF SVMs and neural networks allow curved boundaries. For F8, the neural classifier with seed 42 missed three good observations and incorrectly accepted two bad ones; logistic regression missed five and accepted one. I’d tolerate some wasted evaluations to avoid excluding promising regions, while retaining exploration outside the predicted good area. Thresholding also discards how much better one score is than another.

When choosing a model to guide the search, I balanced interpretability against held-out predictive performance. F3’s Gaussian process had leave-one-out regression RMSE of 0.0757, compared with the selected neural architecture’s mean of 0.1134. F6 favoured RBF support-vector regression: 0.286 against 0.376 for the network. Linear coefficients are easier to inspect, but I wouldn’t accept substantially worse predictions for that convenience. The results favour different models for different functions; the Gaussian process remains my main F3 surrogate and also supplies uncertainty estimates for exploration.

To prioritise further exploration, I compared the neural network’s input gradients across observed locations and three seeds. For F8, the largest average absolute gradients were x3, x1 and x7, approximately 2.12, 1.53 and 1.09. At the best observed input, the network, Gaussian process and support-vector regressor all favoured increasing x3 and decreasing x1 and x7. I’d test small, controlled changes along those coordinates, subject to the bounds. These gradients describe fitted-model sensitivity at sampled locations; they don’t establish global variable importance.

To assess the neural network’s good/bad boundary, I examined held-out classification results and conditional plots. F8 achieved 84.4% leave-one-out balanced accuracy with seed 42, averaging 83.0% across three seeds, versus 77.6% for logistic regression. The seed-42 model classified its training data perfectly, which made me cautious about generalisation. Backpropagation fitted the weights; derivatives with respect to inputs then showed which coordinates changed predicted class probability. The plots held other inputs fixed, so they showed slices of the fitted boundary. We cannot establish how closely those slices match the unknown true boundary.

Comparing neural networks with simpler models, F5 gave the clearest evidence that extra flexibility helped. Mean neural regression RMSE was about 127, versus 1655 for linear regression and 335 for the best tested non-neural alternative, a Gaussian process. F5 was the only function where the selected neural architecture beat the best non-neural regressor. Its 49 parameters served just 24 observations; comparing architectures and seeds added tuning work, and interpretation required gradient analysis. I’d retain it as an F5 cross-check. These retrospective comparisons followed the fifth submission, reused small validation sets for model selection and had no independent final test.

## Supporting evidence

- [Cross-function surrogate comparison](../results/week-05-surrogate-comparison.md)
  records the methods, seed averages and limitations for all eight functions.
- [F3 study](../notebooks/Function_3/Week_05/official/surrogate-study/report.md)
  supports the decision against using the current classifier to guide queries.
- [F5 study](../notebooks/Function_5/Week_05/surrogate-study/report.md) and
  [measured progress](../results/progress.md) distinguish the suspected boundary
  maximum from the confirmed best observed score.
- [F8 study](../notebooks/Function_8/Week_05/surrogate-study/report.md)
  records the classification errors, gradients and conditional plots.

The studies use 207 official observations through round 4 and were run after
the fifth queries were submitted. Posting this reflection does not record new
function evaluations, a marked result or completion of the peer-comment requirement.
