# 08 · Applied Statistical Methods in R (Chi-square → PCA → Regression)

> **MAXD 5133 Applied Statistical Method · Written Assignment · individual · January 2026**
> One script per question, each a complete small analysis.

## 1 · Why this project exists

Most "learn statistics" material teaches the tests. What it rarely teaches is the thing you
actually need on the job: **which test fits which question, and what has to be true before you are
allowed to run it.**

This assignment is that, worked through seven times. It is the page I come back to when I have a
question and need to remember what the right instrument is.

| Script | Method | Data | Use it when |
|---|---|---|---|
| `Q1_Q2_chisquare_nonparametric.R` | Chi-square goodness-of-fit + test of independence | frequency counts | categorical data, no normality assumption |
| `Q3_titanic_data_cleaning.R` | Missing-value diagnosis and imputation (`VIM`, `mice`) | Titanic | before any model, when the data has holes |
| `Q5_anova.R` | One- and two-way ANOVA | poisons | comparing means across 3+ groups |
| `Q6_pca.R` | Principal component analysis | decathlon (10 events) | many correlated numeric columns |
| `Q7_factor_analysis.R` | Factor analysis with rotation (`psych`, `GPArotation`) | cars | looking for latent constructs behind measures |
| `Q8_multiple_regression.R` | Multiple regression + diagnostics | transport | predicting a continuous outcome |
| `Q9_logistic_regression_credit.R` | Logistic regression, full vs reduced | ISLR `Credit` | predicting a yes/no outcome |

---

## 2 · The concepts behind this assignment

### 2.1 The decision tree: which test, and why

Almost every choice comes down to two questions — **what kind of variable is the outcome**, and
**how many groups are you comparing**.

| Outcome | Predictor(s) | Method |
|---|---|---|
| Categorical counts | one variable vs a theoretical distribution | Chi-square goodness of fit |
| Categorical counts | two categorical variables | Chi-square test of independence |
| Continuous | one categorical, 2 groups | t-test |
| Continuous | one categorical, 3+ groups | ANOVA |
| Continuous | two categorical | Two-way ANOVA |
| Continuous | several continuous | Multiple regression |
| Binary | anything | Logistic regression |
| *(none — no outcome)* | many correlated numeric | PCA or factor analysis |

The last row is the one people misfile. PCA and factor analysis have **no outcome variable** at
all — they are unsupervised, and their job is to describe structure, not to predict.

### 2.2 Hypothesis testing, p-values and what they do not mean

Every test in the first half follows the same skeleton:

1. State a **null hypothesis** (H₀) — usually "no effect", "no association", "no difference".
2. Compute a test statistic from the data.
3. Ask: **if H₀ were true, how likely is a statistic this extreme or more?** That probability is
   the **p-value**.
4. Small p — the data are surprising under H₀ — so reject it.

Three things a p-value is **not**, and being clear on these is most of statistical literacy:

- It is **not** the probability that H₀ is true.
- It is **not** a measure of effect size. With enough data, a trivially small effect gets a tiny p.
- p > 0.05 is **not** proof of no effect. It is failure to detect one, which is different.

**Type I error** is rejecting a true H₀ (a false positive, rate α, conventionally 0.05). **Type II
error** is failing to reject a false H₀ (a false negative). Lowering α trades one for the other.

### 2.3 Parametric vs non-parametric

**Parametric** tests (t-test, ANOVA, regression) assume the data follow a particular
distribution — usually normal — and are more powerful when that assumption holds.

**Non-parametric** tests (chi-square on counts, Mann–Whitney, Kruskal–Wallis) make no
distributional assumption. They are safer on messy or ordinal data, and they give up some power in
exchange.

Chi-square sits in the non-parametric family, which is why it is the right tool for frequency
counts where "normally distributed" is not even a meaningful thing to ask.

### 2.4 The chi-square statistic

```text
X² = sum over all cells of  (observed - expected)² / expected
```

The statistic grows as observed counts drift from what the null predicts. Dividing by `expected`
is what makes a discrepancy of 10 in a cell expecting 20 count far more than the same 10 in a cell
expecting 2,000.

Where `expected` comes from is the only difference between the two tests:

- **Goodness of fit** — from a theoretical distribution you supply.
- **Test of independence** — from the row and column totals themselves, under the assumption that
  the two variables are unrelated.

**The assumption to state:** expected counts should be at least 5 in each cell. Below that the
chi-square approximation breaks down and you want Fisher's exact test instead.

### 2.5 ANOVA — why not just run lots of t-tests

With three groups you could run three pairwise t-tests. The problem is **multiple comparisons**:
at α = 0.05 each test has a 5% false-positive rate, so three tests give roughly a 14% chance of at
least one false positive, and it gets worse fast.

ANOVA tests all groups at once with a single α. It compares **between-group variance** to
**within-group variance**:

```text
F = variance between group means / variance within groups
```

If the groups really are the same, both estimate the same underlying variance and F ≈ 1. A large F
means the group means are further apart than the noise within groups can explain.

**Two-way ANOVA** adds a second factor, and — the part that makes it worth doing — can test the
**interaction**: does the effect of the treatment depend on which poison it is? That question does
not exist in a one-way design.

Assumptions: independent observations, roughly normal residuals, and similar variance across
groups (homoscedasticity).

### 2.6 PCA — what it actually does

Principal component analysis finds new axes — **principal components** — that are linear
combinations of the original variables, chosen so that:

- **PC1** captures the most variance possible in one direction;
- **PC2** captures the most of what is left, subject to being **orthogonal** to PC1;
- and so on.

The result is a set of uncorrelated components ordered by how much they explain. Keep the first
few and you have reduced ten correlated columns to two or three that carry most of the
information.

**Why scaling is compulsory** when units differ: PCA maximises *variance*, and variance has units.
Measure one event in seconds (values around 11) and another in metres (values around 800), and the
metres column dominates PC1 by arithmetic accident, not because it carries more information.
`scale. = TRUE` converts everything to z-scores first so each variable gets an equal vote.

### 2.7 PCA vs factor analysis — a genuinely different question

These get confused constantly, and the distinction is conceptual, not technical.

| | PCA | Factor analysis |
|---|---|---|
| Question | how do I compress these variables? | what unobserved thing is *generating* these correlations? |
| Direction | components are built **from** the variables | factors are assumed to **cause** the variables |
| Variance modelled | all of it, including noise | only the variance shared between variables |
| Typical use | dimensionality reduction before modelling | finding latent constructs (ability, attitude, quality) |

**Rotation** — varimax (keeps factors uncorrelated) or oblimin (allows correlation) — is applied
afterwards. It does not change how well the model fits; it rotates the axes so each factor loads
heavily on a few variables and near-zero on the rest. That is purely for interpretability, and it
is what turns "factor 2 is a bit of everything" into "factor 2 is engine size".

### 2.8 Multiple regression, and the diagnostics that are not optional

```text
y = β₀ + β₁x₁ + β₂x₂ + … + ε
```

Each **β** is read as: the expected change in `y` for a one-unit increase in that `x`, **holding
the others constant**. That last clause is the whole point of multiple regression and the part
most often dropped when reporting.

Four assumptions, and what breaks if each fails:

| Assumption | Check | If violated |
|---|---|---|
| **Linearity** | residuals vs fitted should show no pattern | coefficients are biased |
| **Independence** | domain knowledge; Durbin–Watson for time data | standard errors are wrong |
| **Homoscedasticity** | residual spread should be constant | standard errors are wrong, p-values unreliable |
| **Normal residuals** | Q–Q plot | inference is unreliable on small samples |

Plus **multicollinearity**: when predictors are strongly correlated with each other, the
individual coefficients become unstable and uninterpretable even though the overall fit is fine.
The VIF is the standard check.

**This is why the diagnostics are compulsory:** a high R² sitting on a violated assumption is a
wrong answer with a confident number attached, and nothing in the output warns you.

### 2.9 Logistic regression — why linear regression cannot do binary

Fit a straight line to a 0/1 outcome and it will happily predict 1.4 and −0.2, which are not
probabilities. Logistic regression models the **log-odds** instead:

```text
log( p / (1-p) ) = β₀ + β₁x₁ + …
```

The left side can be any real number; the inverse (the logistic function) squashes it back into
(0, 1). Reading the coefficients:

- **β** is a change in log-odds — not intuitive.
- **exp(β)** is an **odds ratio**, which is. `exp(β) = 1.5` means the odds multiply by 1.5 for each
  one-unit increase in that predictor.

In R, `family = binomial` is the single argument that makes `glm()` logistic rather than linear.

**Full vs reduced model comparison** asks whether the extra predictors earn their place. More
predictors always fit the training data better, so the comparison is made on AIC (which penalises
parameter count) or a likelihood-ratio test — not on raw fit.

### 2.10 Missing data: diagnose before you impute

This is the single most important habit in the whole assignment, and it is the same
MCAR/MAR/MNAR distinction that drives the decision in
[04 · Airbnb](../04-airbnb-price-occupancy-knime).

- **MCAR** — missing completely at random. Dropping rows is safe.
- **MAR** — missing at random, given other observed variables. Imputation works if you condition
  on them.
- **MNAR** — missing *because of* the value itself. Imputation will bias the result and there is
  no clean fix inside the dataset.

`VIM` visualises the **pattern** of missingness — is `Age` missing uniformly, or missing mostly for
third-class passengers? That question has to be answered before `mice` runs, because **imputing
before diagnosing is how you manufacture a bias and never notice.**

`mice` itself does *multiple* imputation: it generates several complete datasets reflecting the
uncertainty in the fill-in values, rather than pretending one guess is a fact.

---

## 3 · Q1 · Chi-square goodness of fit

*"Does the observed split match the expected one?"*

```r
observed      <- c(142, 57, 51, 50)
expected_prop <- c(0.53, 0.19, 0.14, 0.14)

chisq.test(observed, p = expected_prop)
```

One categorical variable against a theoretical distribution (2.4). Assumption to state: expected
counts ≥ 5 in each cell.

## 4 · Q2 · Chi-square test of independence

*"Are these two variables related?"*

```r
tornadoes <- matrix(
  c(26,  4, 87,  97,
     2, 41, 46,  63,
    13, 25, 18, 225),
  nrow = 3, byrow = TRUE)
rownames(tornadoes) <- c("January", "February", "March")
colnames(tornadoes) <- c("2015", "2014", "2013", "2012")

chisq.test(tornadoes)
```

Same statistic, different question — the expected counts now come from the row and column totals
themselves (2.4). Reject the null and month and year are **not** independent: tornado counts
depend on which month you are looking at.

## 5 · Q3 · Cleaning the Titanic properly — the long one

This script is a full audit, and the order is deliberate.

**1 · Familiarise.** `str`, `dim`, `head`, `tail`, `summary`, uniqueness of `PassengerId`, plus
survival rates overall, by sex and by class, and histograms of age and fare. **You cannot spot a
wrong value until you know what a right one looks like.**

**2 · Structural checks.** Duplicate ids, duplicate names, and every value range:

```r
unique(titanic$Survived)                        # must be only 0 / 1
unique(titanic$Pclass)                          # must be only 1 / 2 / 3
titanic %>% filter(Age < 0 | Age > 80)          # impossible ages
titanic %>% filter(SibSp < 0 | Parch < 0)       # impossible family counts
titanic %>% filter(Fare < 0)                    # impossible fares
unique(titanic$Sex); unique(titanic$Embarked)   # category typos
```

Three different classes of error are being hunted here: values outside a known set, values outside
a physically possible range, and category typos (`male` / `Male` / `M` as three "categories").

**3 · Logical inconsistencies** — the ones that are *possible* but suspicious: `Age == 0`,
`Fare == 0`, families of more than 7. These are judgement calls, not errors, and they get reported
rather than silently fixed.

**4 · Outliers by IQR, then by eye.**

```r
Q1_fare <- quantile(titanic$Fare, 0.25, na.rm = TRUE)
Q3_fare <- quantile(titanic$Fare, 0.75, na.rm = TRUE)
IQR_fare <- Q3_fare - Q1_fare
fare_outliers <- titanic %>%
  filter(Fare < (Q1_fare - 1.5 * IQR_fare) | Fare > (Q3_fare + 1.5 * IQR_fare))
```

A high fare is not automatically an error — first-class cabins really did cost that much — so the
outliers are **reported, then boxplotted, then judged**. Auto-deleting them would delete the entire
first class, which is a real and important part of the survival story.

**5 · Types.** `Survived` and `Pclass` become factors with labels, so every later model treats them
as categories rather than numbers.

**6 · Missing values with `VIM` and `mice`** — the diagnose-then-impute discipline from 2.10.

## 6 · Q5 · ANOVA — factors first

```r
poisons <- read.csv("https://raw.githubusercontent.com/guru99-edu/R-Programming/master/poisons.csv")
poisons$poison <- factor(poisons$poison)
poisons$treat  <- factor(poisons$treat)

summary(aov(time ~ poison + treat, data = poisons))
```

`poison` arrives as 1/2/3. Leave it numeric and R fits a **regression on the number** — asking
"does survival time increase linearly with poison ID?", which is meaningless. The `factor()` calls
*are* the analysis.

`time ~ poison + treat` is a two-way ANOVA testing both factors (2.5). Writing `poison * treat`
instead would add the interaction term.

## 7 · Q6 · PCA — scale, or the biggest unit wins

```r
decathlon    <- read.csv("../data/decathlon.csv")
athlete_name <- decathlon[, 1]                       # names out of the numeric matrix
pca <- prcomp(decathlon[, 2:11], scale. = TRUE)      # seconds and metres are not comparable
```

Ten events measured in seconds, metres and points — the textbook case for 2.6. Without
`scale. = TRUE`, the event with the largest numeric range becomes PC1 by arithmetic accident.

Decathlon is a good PCA dataset precisely because the correlations are real and interpretable: PC1
usually separates overall athletic ability, and PC2 the sprint/endurance or speed/strength
trade-off.

## 8 · Q7 · Factor analysis — a different question from PCA

```r
library(psych); library(GPArotation)
cars <- read.csv("../data/cars.csv")
```

PCA compresses; factor analysis asks *what unobserved thing is generating these correlations*
(2.7). Rotation via `GPArotation` then makes each factor load on a few variables instead of a
little of everything, which is what makes it interpretable.

## 9 · Q8 · Multiple regression — the model plus the diagnostics

```r
transport <- read.csv("../data/transport_dataset.csv")
model <- lm(travel_time ~ ., data = transport)
summary(model)
```

`travel_time ~ .` means "regress on every other column". The question asked for the **regression
equation**, which means coefficients with units and a sentence per coefficient — each one read
with the "holding the others constant" clause from 2.8.

Then residual plots, because a good R² over a violated assumption is a wrong answer with a
confident number attached.

## 10 · Q9 · Logistic regression — full vs reduced

```r
library(ISLR)
credit_data <- Credit
credit_data$Married <- ifelse(credit_data$Married == "Yes", 1, 0)   # Yes/No -> 1/0

model_full <- glm(Married ~ ., data = credit_data, family = binomial)
summary(model_full)
```

`family = binomial` is what makes this logistic rather than linear (2.9): the model predicts
log-odds, so coefficients are read as odds ratios once exponentiated. The assignment then compares
the full model to a reduced one — fewer predictors, and the question of whether the extra ones
earned their place.

---

## 11 · The five habits this assignment drilled

1. **Convert grouping variables to factors** before any group comparison. R will not warn you.
2. **Scale before PCA** whenever units differ (2.6).
3. **Diagnose missingness before imputing** (2.10), and let the pattern decide whether imputation
   is honest at all.
4. **Report outliers, do not auto-delete them** — a true extreme value is data, not dirt.
5. **A fitted model without diagnostics is half an answer** (2.8).

## 12 · How this connects to the other projects

- 2.10 is the same MCAR/MAR/MNAR reasoning that made me drop rather than impute the review-score
  columns in [04 · Airbnb](../04-airbnb-price-occupancy-knime).
- The IQR discussion in Q3 is the same rule applied more carefully in Airbnb, where the outlier
  cut-off had to be computed **per room type** rather than globally.
- 2.2's "p is not effect size" is the statistical cousin of the accuracy-vs-F1 argument in
  [03 · E-commerce](../03-ecommerce-purchase-prediction): a number can be technically correct and
  still answer the wrong question.

## Files

```text
R/     the seven scripts — absolute Windows paths rewritten to ../data/... so they run from a clone
data/  the small datasets they read
```
