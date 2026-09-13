# 03 · E-Commerce Purchase Prediction & Customer Segmentation

> **MAXD 5143 Applied Machine Learning · Assignment 2 · individual · June 2026**
> [Colab notebook](https://colab.research.google.com/drive/1iuEBPeCRgzTBAdODdMPm5W2qQ1Ey7coj)

## 1 · Why this project exists

Case study **"Global-Mart"**, an online retailer sitting on piles of transactional data and no
idea what to do with it. The brief asks for a three-tier ML system that answers three different
commercial questions, plus a fourth on basket analysis:

| Task | The business question | The ML question |
|---|---|---|
| **1 · Segmentation** | Who are our customers, really? | unsupervised — how many natural groups exist? |
| **2 · Prediction** | Who will respond to a high-value promotion? | supervised classification on an imbalanced target |
| **3 · Ensembles** | Can we do better than one model? | bagging vs boosting vs voting |
| **4 · Market basket** | What gets bought together? | association rule mining |

The constraint that shaped everything: **every task had to be an experiment.** Compare at least
two methods, justify the choice with a metric, and back it with a journal reference. No "I used
Random Forest because it's good."

---

## 2 · The concepts behind this project

Four different families of technique appear here. This section explains each one before it is
used, because the results only make sense if you know what the method is assuming.

### 2.1 Supervised vs unsupervised learning

- **Unsupervised** (Task 1): there are no labels. You are asking the data "what structure is
  already in you?" There is no right answer to check against, which is why you need *internal*
  validation measures like silhouette (2.4).
- **Supervised** (Tasks 2 and 3): every row has a known answer. The model learns a mapping from
  features to that answer, and you measure how well it does on rows it has never seen.
- **Association rule mining** (Task 4) is neither — it is pattern discovery, closer to a
  descriptive statistic than a model.

### 2.2 Why standardisation is compulsory before clustering

Annual spend runs in thousands; visit frequency runs in tens. Clustering measures **distance**
between points, usually Euclidean:

```text
d = sqrt( (spend_a - spend_b)^2 + (freq_a - freq_b)^2 )
```

A 2,000-unit difference in spend swamps a 20-visit difference in frequency completely. Without
standardising, the algorithm is effectively clustering on spend alone and the second feature does
nothing.

`StandardScaler` fixes this by converting each feature to **z-scores**: subtract the mean, divide
by the standard deviation, so every feature has mean 0 and standard deviation 1 and contributes
equally to the distance.

### 2.3 How K-Means actually works

1. Pick *k* starting centroids.
2. Assign every point to the nearest centroid.
3. Move each centroid to the mean of the points assigned to it.
4. Repeat 2–3 until nothing moves.

Two properties that matter in practice:

- **It converges to a local optimum, not a global one.** Different starting centroids give
  different answers. That is why the code uses `n_init=10` — run it ten times from different
  starts and keep the best.
- **You must choose *k* yourself.** The algorithm cannot tell you how many groups exist, which is
  the entire reason 2.4 exists.

### 2.4 Choosing *k*: silhouette and the elbow

**Silhouette score**, for each point:

```text
a = mean distance to other points in its own cluster
b = mean distance to points in the nearest other cluster
s = (b - a) / max(a, b)
```

The score runs from −1 to +1. Near **+1** means the point is much closer to its own cluster than
to any other — good. Near **0** means it sits on a boundary. **Negative** means it is probably in
the wrong cluster. The score for a clustering is the average over all points.

**Inertia / the elbow method**: inertia is the total squared distance from each point to its
centroid. It always falls as *k* rises — with *k* = *n*, inertia is zero and the clustering is
worthless. So you do not minimise it; you look for the **elbow**, the point where adding another
cluster stops buying much reduction.

Silhouette is judgement-free and gives a number. The elbow is a visual heuristic. **Using both
and getting the same answer is much stronger evidence than either alone**, and that is exactly
what happened here.

### 2.5 Hierarchical clustering and the linkage choice

Agglomerative clustering starts with every point as its own cluster and repeatedly merges the two
closest clusters. "Closest" needs a definition, and that definition is the **linkage**:

| Linkage | Distance between clusters = | Behaviour |
|---|---|---|
| **Single** | distance between the two *nearest* members | produces long straggly chains; one line of noise points welds two real clusters together |
| **Complete** | distance between the two *farthest* members | produces compact, roughly equal-diameter clusters; resistant to chaining |
| Average | mean of all pairwise distances | a compromise |

The chaining failure of single linkage is not theoretical — Figure 1 shows stray points sitting
in the gap between the two customer groups, and Figure 3 shows single linkage scoring worse
because of them.

### 2.6 Class imbalance, and why accuracy lies

84.53% of sessions in the Task 2 dataset do not end in a purchase. A model that predicts "no
purchase" for every single session — a model containing no intelligence at all — scores **84.53%
accuracy**.

So on imbalanced data you report:

```text
Precision = TP / (TP + FP)   of the sessions I flagged, how many really bought?
Recall    = TP / (TP + FN)   of the real buyers, how many did I catch?
F1        = 2PR / (P + R)    harmonic mean, punishes lopsidedness
```

And a **confusion matrix**, which is just the four raw counts laid out in a square. Every
aggregate metric above is computed from those four numbers, which is why the confusion matrices
in section 7 are the real evidence and the accuracy bar chart is decoration.

### 2.7 Stratification and leakage

**Stratified splitting** preserves the class ratio in both halves. Without it, random chance can
concentrate the rare class on one side, and you end up evaluating on a test set that does not
look like the training set.

**Leakage** is letting information from the test set influence training. The version that bites
everyone: fitting the scaler on all the data before splitting, so the training process has seen
the test set's mean and standard deviation. The fix is one line — `fit_transform` on train,
`transform` only on test — and it is in the code for exactly this reason.

### 2.8 Bias, variance, and what ensembles fix

Every model's error decomposes into:

- **Bias** — error from the model being too simple to capture the real pattern. Underfitting.
- **Variance** — error from the model being too sensitive to the particular training rows.
  Overfitting.
- **Irreducible noise** — the part nobody can fix.

This matters because **the two ensemble families attack different halves**:

| Family | Example | How it trains | What it reduces |
|---|---|---|---|
| **Bagging** | Random Forest | many models in **parallel** on bootstrap samples, then average | **variance** |
| **Boosting** | AdaBoost | models in **sequence**, each weighting the examples the last one got wrong | **bias** |

A single decision tree is famously high-variance: change a few rows and the splits change
completely. That diagnosis is why Random Forest gave the biggest jump in this project, and it is
the answer to "which ensemble should I use?" — **it depends which problem you have.**

**Voting** is different again: it combines *different model types*. Hard voting counts votes; soft
voting averages predicted probabilities and can be weighted, so a strong model counts for more
than a weak one.

### 2.9 Association rules: support, confidence, lift

For a rule **A → B** ("people who buy A also buy B"):

```text
Support(A→B)    = P(A and B)          how often the pair appears at all
Confidence(A→B) = P(B | A)            of baskets with A, how many have B
Lift(A→B)       = P(B | A) / P(B)     how much more likely than chance
```

**Lift is the one that matters.** A rule can have 90% confidence purely because B is in 90% of all
baskets — that rule tells you nothing. Lift divides the confidence by B's baseline rate, so:

- **lift > 1** — A genuinely makes B more likely
- **lift = 1** — independent, no relationship
- **lift < 1** — A makes B *less* likely

**Apriori** is the algorithm that finds the frequent itemsets efficiently, using one clever
observation: if an itemset is infrequent, every superset of it must also be infrequent, so you
can prune enormous parts of the search space without checking them.

---

## 3 · The thread running through all four tasks

One idea keeps coming back, and it is the thing I would actually take to a job: **pick the metric
before you pick the model.** On this data the same results tell completely opposite stories
depending on whether you report accuracy or F1 — and the assignment is built to make you trip
over that at least once.

---

# Task 1 · Behavioural segmentation

## Step 1 — Build the dataset

The brief said to generate it: 300 customer records with two features, **annual spend** and
**visit frequency**, built so natural groups exist.

![Annual spend against visit frequency](figures/01-customer-distribution.png)

*Figure 1 — All 300 customers. Before any algorithm runs, you can already see two blobs with a gap
between them. That eyeball read is the first evidence that k = 2 is the honest answer — and it is
what the two methods below have to agree with before I trust them. Note also the stray points
sitting inside the gap: those are what break single linkage in Figure 3.*

## Step 2 — Standardise

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data[['Annual_Spend', 'Visit_Frequency']])
```

See 2.2. Skip this and "spend" alone decides every cluster.

## Step 3 — K-Means across k = 2…10, scored two ways

```python
silhouette_scores_kmeans, inertia_values = [], []
k_range = range(2, 11)                    # silhouette is undefined at k = 1

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    silhouette_scores_kmeans.append(silhouette_score(X_scaled, kmeans.labels_))
    inertia_values.append(kmeans.inertia_)      # for the elbow curve
```

`n_init=10` is the local-optimum guard from 2.3. `random_state=42` makes the run reproducible.
`range(2, 11)` starts at 2 because silhouette needs at least two clusters to have a "nearest
other cluster" to measure against.

![K-Means silhouette and inertia elbow](figures/02-kmeans-silhouette-elbow.png)

*Figure 2 — Silhouette (left) and the inertia elbow (right). Silhouette peaks at 0.63 at k = 2 and
falls away; inertia drops sharply from k=2 to k=3 then flattens. Two independent criteria landing
on the same k is worth far more than one criterion you tuned until it agreed with you (2.4).*

## Step 4 — Hierarchical clustering, single vs complete linkage

```python
for k in k_range:
    agg_single = AgglomerativeClustering(n_clusters=k, linkage='single')
    silhouette_scores_agg_single.append(silhouette_score(X_scaled, agg_single.fit_predict(X_scaled)))

    agg_complete = AgglomerativeClustering(n_clusters=k, linkage='complete')
    silhouette_scores_agg_complete.append(silhouette_score(X_scaled, agg_complete.fit_predict(X_scaled)))
```

| Linkage | Best silhouette | At k | Why |
|---|---|---|---|
| Complete | **0.63** | 2 | merges on the *farthest* pair, so clusters stay compact and outliers cannot drag a merge |
| Single | 0.59 | 3 | merges on the *nearest* pair → chaining: a line of noise points welds two real clusters together |

![Single against complete linkage](figures/03-agglomerative-linkage-comparison.png)

*Figure 3 — Complete linkage is above single almost everywhere. Look back at Figure 1 and the
reason is visible: there are stray points sitting in the gap between the two blobs, and single
linkage chains straight through them, welding the two real groups into one. This is 2.5 happening
on real data.*

![Final cluster assignments](figures/04-cluster-assignments.png)

*Figure 4 — K-Means beside agglomerative complete linkage. Two completely different algorithms —
one centroid-based and iterative, one merge-based and deterministic — landing on essentially the
same split: low spend / low frequency versus high spend / high frequency. That agreement is the
result, more than either score on its own.*

**What Global-Mart does with this:** two segments, not five. A "come back" campaign for the
low-frequency group and a loyalty play for the high-value group. Inventing five personas from
this data would be making things up, and the silhouette curve is the evidence.

---

# Task 2 · Purchase prediction

## Step 1 — Real data this time, and it is lopsided

**UCI Online Shoppers Purchasing Intention**: **12,330 sessions**, with page values, bounce rates,
durations, month and visitor type. Target: did the session end in a purchase. No missing values,
so no imputation needed.

![Target distribution](figures/05-revenue-class-imbalance.png)

*Figure 5 — The single most important chart in the assignment. 84.53% of sessions do not buy. A
model predicting "nobody buys" for every session scores 84.53% accuracy and is worth exactly
nothing (2.6). This figure is why precision, recall and F1 on the buying class are reported
everywhere below.*

## Step 2 — Split first, scale second

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)   # keep the 85/15 ratio in both

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)          # fit on train only
X_test_scaled  = scaler.transform(X_test)               # test is only transformed
```

Two habits in three lines, both from 2.7: **stratify**, so the rare class is not accidentally
concentrated on one side, and **fit the scaler on train only**, so nothing from the test set leaks
into training.

## Step 3 — Three classifiers, three different philosophies

```python
knn           = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
decision_tree = DecisionTreeClassifier(random_state=42).fit(X_train_scaled, y_train)
naive_bayes   = GaussianNB().fit(X_train_scaled, y_train)
```

| Model | How it decides | Accuracy | Recall | F1 |
|---|---|---|---|---|
| KNN (k=5) | looks at the 5 nearest sessions and takes a vote | **86.78%** | — | — |
| Decision Tree | a sequence of yes/no splits on single features | 85.28% | 54.97% | **53.64%** |
| Gaussian Naive Bayes | Bayes' rule, assuming features are independent | 77.94% | **67.28%** | — |

The confusion matrices are where that table stops being abstract:

![KNN confusion matrix](figures/06-confusion-matrix-knn.png)

*Figure 6 — KNN. 1,997 non-buyers correct and only 87 false alarms — but 239 buyers missed against
143 caught. Best accuracy, worst at the job you actually care about.*

![Decision Tree confusion matrix](figures/07-confusion-matrix-decision-tree.png)

*Figure 7 — Decision Tree. 210 buyers caught against 172 missed. Slightly lower accuracy than KNN,
meaningfully better at the thing the promotion depends on.*

![Naive Bayes confusion matrix](figures/08-confusion-matrix-naive-bayes.png)

*Figure 8 — Naïve Bayes. 257 buyers caught — the most of the three — but paid for with 419 false
positives. Whether that is a good trade depends entirely on what a wasted promotion costs versus
a missed sale. That is a finance question, not a modelling one.*

![Accuracy comparison](figures/09-model-accuracy-comparison.png)

*Figure 9 — Accuracy across the three. KNN on top, Naïve Bayes clearly last.*

![F1 comparison](figures/10-model-f1-comparison.png)

*Figure 10 — F1 across the same three, and the ranking flips. Put Figures 9 and 10 side by side in
front of a stakeholder and you have explained imbalanced classification without saying the word
"imbalance".*

### The three discussion questions, answered

1. **Why does KNN need standardisation?** It classifies by distance (2.2). An unscaled feature
   with a big range dominates every distance calculation and the other features stop mattering.
   For KNN this is not a refinement — it is the difference between working and not.
2. **Why does Naive Bayes underperform here?** It assumes features are conditionally independent
   given the class. Session features obviously are not: page values, durations and bounce rates
   all move together, because they are all downstream of how engaged the visitor was. The
   independence assumption is violated, the probabilities get distorted, and the model
   over-predicts the positive class — which is exactly the 419 false positives in Figure 8.
3. **Why is accuracy a trap?** See Figure 5 and 2.6.

---

# Task 3 · Ensemble strategies

## Bagging — Random Forest

```python
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_scaled, y_train)
```

Bagging trains 100 trees, each on a **bootstrap sample** (drawn with replacement, so each tree
sees a slightly different dataset) with a **random subset of features** considered at each split.
Then it averages them.

Why that works, in one sentence: the trees make *different* mistakes, so averaging cancels the
noise while the real signal — which all of them see — survives. This is variance reduction (2.8),
and a single decision tree is the textbook high-variance model, so it is the right medicine.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **89.98%** | 73.20% | 55.76% | **63.30%** |
| AdaBoost | 89.01% | — | — | — |
| Decision Tree (Task 2) | 85.28% | — | 54.97% | 53.64% |

**+4.7 percentage points of accuracy and +9.7 points of F1 over the single tree** — the clearest
result in the assignment.

![Random Forest confusion matrix](figures/11-confusion-matrix-random-forest.png)

*Figure 11 — 2,006 / 78 / 169 / 213. Compare with Figure 7: fewer false alarms **and** more buyers
caught than the single tree. Bagging improved both sides at once, which is not always what
happens — usually you trade one for the other.*

## Boosting — AdaBoost

```python
adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
```

Boosting is **sequential, not parallel**. Each new weak learner is trained with more weight on the
examples the previous ones got wrong, and the final prediction is a weighted vote where
better-performing learners count for more. It attacks **bias** where bagging attacks **variance**
(2.8).

Knowing which of those two problems you actually have is the entire point of the question.

![AdaBoost confusion matrix](figures/12-confusion-matrix-adaboost.png)

*Figure 12 — 1,978 / 106 / 165 / 217 — marginally better buyer recall than Random Forest,
marginally more false alarms. Different mechanism, nearly identical outcome, which is itself
informative: it suggests the remaining error on this dataset is no longer a variance problem, so
neither medicine helps much more.*

## Voting

```python
hard_voting = VotingClassifier(
    estimators=[("KNN", knn), ("Decision Tree", decision_tree), ("Naive Bayes", naive_bayes)],
    voting="hard")
```

Hard voting counts votes; soft voting averages probabilities and can be weighted. With base
learners of unequal quality — Naive Bayes is clearly the weakest here — weighting is what stops
the weak model from cancelling a correct majority. With three models and one bad one, hard voting
gives the bad model a full third of the decision, which is not what you want.

![Ensemble accuracy comparison](figures/13-ensemble-accuracy-comparison.png)

*Figure 13 — Random Forest, AdaBoost, hard voting and soft voting side by side. The ensembles beat
the single models, but the four of them sit within about two points of each other. Worth saying
plainly rather than overselling: the gain came from using an ensemble at all, not from picking
the perfect one.*

---

# Task 4 · Market basket analysis

**7,501 transactions, maximum 20 items each.** Apriori finds the frequent itemsets, then
association rules are scored by support, confidence and lift (2.9).

![Association rules table](figures/14-association-rules-table.png)

*Figure 14 — The rules ranked by lift, with support and confidence alongside. Note how low the
supports are — around 1–2%. These are real patterns, but they are not store-wide behaviours, and
that distinction changes what you are allowed to recommend.*

![Top 10 rules by lift](figures/15-top10-rules-by-lift.png)

*Figure 15 — Top 10 rules by lift. The strongest is **herb & pepper → ground beef, lift 3.29**:
buyers of that seasoning are more than three times likelier than average to buy ground beef. That
is a shelf-placement decision, and a cheap one.*

![Support against confidence](figures/16-support-vs-confidence.png)

*Figure 16 — Support against confidence for every surviving rule. The high-support,
high-confidence corner is empty. The honest conclusion is that this basket data yields niche
cross-sell rules, not a store-wide restructuring — and saying so is more useful than dressing up
a 1.5% support rule as a strategy.*

---

## What I would say about this project in an interview

1. **Pick the metric before the model.** Figures 9 and 10 are the same three models ranked in
   opposite orders. I can explain that in thirty seconds with those two charts.
2. **Two validation methods beating in the same direction** — silhouette *and* elbow, K-Means
   *and* hierarchical — is worth more than one method tuned until it agreed with me.
3. **Bagging vs boosting is a diagnosis, not a preference.** Variance problem or bias problem.
   Here the answer was variance, which is why Random Forest gave the biggest jump.
4. **The 0.5 threshold is a business choice.** Missing a buyer and pestering a non-buyer do not
   cost the same, and nothing in scikit-learn knows that.

## How this connects to the other projects

- [05 · IMDB](../05-imdb-sentiment-nlp) is the mirror image on metrics: balanced classes there,
  so accuracy is fine. Same metric, opposite verdict, because of the data.
- [04 · Airbnb](../04-airbnb-price-occupancy-knime) uses boosting for the opposite reason —
  there the problem is a non-linear relationship (bias), not unstable trees (variance).
- [09 · Power BI](../09-superstore-powerbi) is the same "the headline number hides the thing that
  matters" lesson, arriving from the dashboard side instead of the modelling side.

## Files

```text
notebook.ipynb               the submitted notebook
code/ecommerce_models.py     code cells extracted, markdown kept as comments
figures/                     the 16 figures above, from the submitted report
```
