# 2024 cohort transfer review

[Executed analysis notebook](historical_cohort_review.ipynb) · [Pinned 2024 competition archive](https://github.com/wrm65/Capstone-Competition-2024/tree/27e7ed89734766f89ae23af1871deebb34f3bca4)

The collaborative archive adds **874 rows** to the 25 November current pool,
for **1,364 rows** across the eight functions. It contributes **689 returned
measurements** from nine reporting IDs and **185 additional starter points**.
The archive's first starter set matches all sixteen current starter arrays
byte-for-byte. Each of its combined starter arrays equals the concatenation of
the two published sets. The first set is already represented in the current
pool.

The importer reconciles 102 named query/observation file pairs. It removes
3,593 recurrences of a measurement across accepted cumulative snapshots.
One malformed file pair was skipped. A later valid snapshot contains that
participant's history. Seven function-level measurements at one participant's
cumulative position disagree between copies and are quarantined. The file
audit retains the conflicting alternatives and the path of each imported row.
Repeated evaluations at the same point across distinct positions or
participants are retained as measurements.

## Measured bests

The best checked column includes the 2024 archive, the current pool and
additional published campaigns with matching starter data. It uses paired
measurements at inputs in the current portal range `0 ≤ x < 1`. The archive
covers the publicly reported data available from the checked sources. The
2024 leaderboard ranks 36 entrants, while the repository has paired query
files from nine. Its separate HTML summary reproduces scores already found in
the paired data or shared starters. The leaderboard supplies ranks only, so
the best checked scores are incomplete as an all-cohort record.
[2024 rank table](https://github.com/wrm65/Capstone-Competition-2024/blob/27e7ed89734766f89ae23af1871deebb34f3bca4/docs/images/leadboard_ranking.jpg)
· [2024 score summary](https://github.com/wrm65/Capstone-Competition-2024/blob/27e7ed89734766f89ae23af1871deebb34f3bca4/docs/html/function_data.html)

| Function | Your best | 2024 best | Best checked | Gap to your best | Best checked input |
| --- | ---: | ---: | ---: | ---: | --- |
| F1 | 1.997355699718 | 1.999999999676 | 1.999999999676 | 0.002644300 | `0.628540-0.628540` |
| F2 | 0.723057545575 | 0.919871693233 | 0.919871693233 | 0.196814148 | `0.500000-0.500000` |
| F3 | −0.004952000822 | −0.000253026028 | −0.000140306768 | 0.004811694 | `0.520000-0.550000-0.450000` |
| F4 | 0.666089598934 | 0.692066924633 | 0.712145930513 | 0.046056332 | `0.407459-0.418058-0.364200-0.420348` |
| F5 | 8662.405001248 | 8662.405001248 | 8662.405001248 | 0 | `0.999999-0.999999-0.999999-0.999999` |
| F6 | −0.213959540335 | −0.006143582423 | −0.006143582423 | 0.207815958 | `0.424197-0.368357-0.648904-0.762441-0.113934` |
| F7 | 2.739375118004 | 3.319276412465 | 3.322368011393 | 0.582992893 | `0.201689-0.150011-0.476874-0.275332-0.311652-0.657301` |
| F8 | 9.936773424663 | 9.999943700473 | 10.000000000000 | 0.063226575 | `0.100000-0.150000-0.130000-0.150000-0.800000-0.500000-0.200000-0.600000` |

The 2024 result files support the strongest checked returns for F1, F2, F5
and F6. [F1](https://github.com/wrm65/Capstone-Competition-2024/blob/27e7ed89734766f89ae23af1871deebb34f3bca4/data/week_01/428-observations.txt)
· [F2](https://github.com/wrm65/Capstone-Competition-2024/blob/27e7ed89734766f89ae23af1871deebb34f3bca4/data/week_02/431_observations.txt)
· [F5](https://github.com/wrm65/Capstone-Competition-2024/blob/27e7ed89734766f89ae23af1871deebb34f3bca4/data/week_10/428-observations.txt)
· [F6](https://github.com/wrm65/Capstone-Competition-2024/blob/27e7ed89734766f89ae23af1871deebb34f3bca4/data/week_04/464-observations.txt)

The November pool supplies the leading F3 and F4 measured scores.
[F3 published query history](https://github.com/originalaby/imperial-ml-capstone/blob/b72a6be9283bb874a8d4b9856ba6f5d352da3375/data/queries.csv)
· [F4 published notebook](https://github.com/kosina/bbocapstone/blob/8db82a121efe015d94cb01d19d1439adf21287a7/capstone_bbo.ipynb)

An independently published campaign reports the F7 and F8 records as weekly
measurements. It also reports 8662.4825 for F5 at the all-1.0 input, outside
the current portal range as published. This campaign was checked for reference
scores only. [Independent campaign's observation table](https://github.com/Wizen-Labs/you-sank-my-function-bbo-challenge/blob/32a026641895299fce5b4250def26a564a078cea/report/consolidated_observations.csv)

## What the additional rows change

- **F1.** The archive's best coordinate is 0.000742 from your incumbent and
  confirms the strongest measured part of the narrow peak. It is closer to
  your best point than the positive-source proposal in the Week 10 batch.
- **F2.** Eight archived evaluations at (0.5, 0.5) ranged from 0.75571 to
  0.91987, with median 0.79325. All eight exceed your current best. The
  current pool has no evaluation at that exact coordinate.
- **F3.** The archive adds near-zero scores around x1 = 0.52 and x2 = 0.53.
  The strongest in the current pool remains the published point at
  (0.52, 0.55, 0.45).
- **F4.** The archived 0.69207 and current-peer 0.71215 both support the
  high-scoring region around x2 = 0.37 to 0.42 and x3 near 0.37.
- **F5.** The archive reproduces your upper-corner return at portal precision.
- **F6.** Four archived returns within 0.035 of the best input ranged from
  −0.06481 to −0.00614. The current pool has no point within that radius.
  This is a different neighbourhood from the x5 = 0 face.
- **F7.** The 2024 results above 3.3 cluster with x1 near 0.21 to 0.23.
  The independent result at x1 = 0.201689 reinforces that region.
- **F8.** The 2024 queries independently reached 9.99994 near the setting
  at which the other campaign measured 10.

## Retrospective prediction check

Each fit held out one of the nine current official returns and removed every
training measurement at that six-decimal input. Repeated training coordinates
were collapsed to their median. Hyperparameters were held fixed across both
training scopes. F1 uses signed-log MAE. Other rows use raw-output RMSE.

| Function | Matérn GP, current → +2024 | 5-NN, current → +2024 |
| --- | ---: | ---: |
| F1 | 66.5459 → 77.1583 | 22.8865 → 63.5724 |
| F2 | 0.108190 → 0.114881 | 0.144458 → 0.118385 |
| F3 | 0.041037 → 0.041918 | 0.027797 → 0.041015 |
| F4 | 0.266918 → 0.229361 | 1.203679 → 1.163874 |
| F5 | 325.836 → 94.6519 | 335.428 → 375.654 |
| F6 | 0.094840 → 0.101819 | 0.312750 → 0.215165 |
| F7 | 0.095889 → 0.030548 | 0.154862 → 0.154788 |
| F8 | 0.524342 → 0.415545 | 0.456334 → 0.490329 |

The archive lowered error for the GP on F4, F5, F7 and F8. Five-nearest
neighbours improved on F2, F4 and F6, with little change on F7. The comparison
uses nine targets per function. The observed historical maxima and their
nearby measurements are especially useful when the current observations
cover a different region from the archive.

## Reproduce the review

From the Capstone root, use the project's local Python environment and a
local Git repository containing the pinned commit.

~~~powershell
git init .runtime/historical-cohort/wrm65
git -C .runtime/historical-cohort/wrm65 fetch --depth 1 https://github.com/wrm65/Capstone-Competition-2024.git 27e7ed89734766f89ae23af1871deebb34f3bca4
.\run-python.ps1 stage-2-bbo/side-quests/historical-cohort-2024/build_snapshot.py --archive .runtime/historical-cohort/wrm65
~~~

Restore the [25 November cohort data](../../data/README.md#full-cohort-refresh-on-25-november-2026)
if needed. Run `historical_cohort_review.ipynb` with the Capstone `.venv`
kernel and the notebook's own directory as its working directory. It
verifies both input snapshots and writes the numerical analysis to the
Git-ignored snapshot directory with the source file fingerprints.
