# Data

Downloaded from the official DrivenData competition page on 29 July 2026.

| File | Purpose | Rows | Columns | Bytes | SHA-256 |
|---|---|---:|---:|---:|---|
| `TrainingSetValues.csv` | Training predictors | 59,400 | 40 | 20,069,199 | `D8EBE40F49FE749A851C8EA28601115CD92442F7B20025FBC33881595AE75F5D` |
| `TrainingSetLabels.csv` | Training `status_group` labels | 59,400 | 2 | 1,148,327 | `AE9B4F893E8E89DF3A2187D38CADE75C61DFB1D1E156ECD7615FEE14FDBE0A24` |
| `TestSetValues.csv` | Test predictors | 14,850 | 40 | 5,016,337 | `A222110D5606910953607EFA5112EAFB1D6C30A483C4CBCD0B92C8306125C9B5` |
| `SubmissionFormat.csv` | Submission template | 14,850 | 2 | 324,512 | `387B15F692DA6196B1F2610BE1AA327A8CBE8DFFA15454B25882F41A4077B9F6` |

The 40 value columns comprise `id` plus the 39 predictors documented in
`../instructions/competition-reference.md`. Verification found no identifier
differences between the training values and labels, or between the test values
and submission template.

## Training target balance

| `status_group` | Rows | Share |
|---|---:|---:|
| `functional` | 32,259 | 54.3% |
| `non functional` | 22,824 | 38.4% |
| `functional needs repair` | 4,317 | 7.3% |

## Licence and handling

The authenticated download page viewed on 29 July 2026 included attribution
guidance. The public competition rules checked on 15 August 2026 require
participants to prevent access by people who have not accepted the rules and
forbid redistribution unless the competition site grants an exception. This
repository therefore treats the supplied CSVs as competition-restricted. Do not
commit them. The Pump It Up `.gitignore` excludes the files while retaining this
inventory.
