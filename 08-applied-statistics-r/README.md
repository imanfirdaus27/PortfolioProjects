# Applied Statistical Methods in R

**Applied Statistical Method (MAXD 5133), Written Assignment, UTeM · January 2026 · individual**

One script per question, each a small, complete analysis in R.

| Script | Method | Data |
|---|---|---|
| `Q1_Q2_chisquare_nonparametric.R` | Chi-square goodness-of-fit and nonparametric tests | frequency counts |
| `Q3_titanic_data_cleaning.R` | Missing-value diagnosis and imputation (`VIM`, `mice`), cleaning pipeline | Titanic |
| `Q5_anova.R` | One- and two-way ANOVA with factor conversion | poisons |
| `Q6_pca.R` | Principal component analysis on 10 event scores | decathlon |
| `Q7_factor_analysis.R` | Factor analysis with rotation (`psych`, `GPArotation`) | cars |
| `Q8_multiple_regression.R` | Multiple regression, model equation and diagnostics | transport |
| `Q9_logistic_regression_credit.R` | Logistic regression, full and reduced models | ISLR `Credit` |

## Files
```
R/       the seven scripts (file paths made relative so they run anywhere)
data/    the small datasets they read
```
Run any script from inside `R/` with `Rscript` or RStudio.
