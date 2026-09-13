# IMDB Movie Review Sentiment Analysis

**Big Data Analytics and Visualization (MAXD 5153), Assignment 1, UTeM · May 2026 · pair**
M. Iman Firdaus and M. Syahmi. Written up in JACTA journal format.

## Data
The IMDB review set: **50,000 labelled reviews**, balanced positive/negative.

## Pipeline
1. **EDA** - class balance, review length, word counts, most common words, word cloud.
2. **Preprocessing** - lowercasing, punctuation and HTML stripping, stopword removal,
   lemmatisation (NLTK `WordNetLemmatizer`) and **negation handling**, so "not good" stops
   being read as "good".
3. **Features** - TF-IDF with n-grams; PCA only for visualising the feature space.
4. **Models** - Logistic Regression, Multinomial Naive Bayes, Linear SVM.
5. **Tuning** - `GridSearchCV` with cross-validation; evaluation on accuracy, precision,
   recall, F1 and the confusion matrix.

## Result
**Logistic Regression was best at about 88% accuracy**, with Linear SVM close behind. On
sparse high-dimensional TF-IDF features a linear model is hard to beat, and it stays
interpretable: the largest coefficients are the words that swing a review.

## Files
```
notebook.ipynb                the submitted notebook
code/sentiment_pipeline.py    code cells extracted, markdown kept as comments
```
Dataset: [IMDB 50K reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (not included).

## Figures

The 8 images below are the real figures from the MAXD 5153 Assignment 1 report, extracted from the submitted PDF.

![Word cloud after cleaning — a sanity check that the pipeline worked.](figures/01-wordcloud-after-cleaning.png)

*Word cloud after cleaning — a sanity check that the pipeline worked.*

![Most frequent tokens across the corpus.](figures/02-top-words.png)

*Most frequent tokens across the corpus.*

![Top words split by sentiment. The overlap is why bag-of-words caps near 88%.](figures/03-top-words-positive-negative.png)

*Top words split by sentiment. The overlap is why bag-of-words caps near 88%.*

![TF-IDF projected to two PCA components — separable, not cleanly.](figures/04-tfidf-pca-projection.png)

*TF-IDF projected to two PCA components — separable, not cleanly.*

![Logistic regression — 88.84% validation, best of three.](figures/05-logistic-regression-report.png)

*Logistic regression — 88.84% validation, best of three.*

![Linear SVM — 88.44%. Close enough that LR won on cost and interpretability.](figures/06-svm-report.png)

*Linear SVM — 88.44%. Close enough that LR won on cost and interpretability.*

![Held-out test: 88.17%, almost identical to validation.](figures/07-final-test-report.png)

*Held-out test: 88.17%, almost identical to validation.*

![Test confusion matrix: 10,947 / 1,553 / 1,404 / 11,096 — errors balanced across classes.](figures/08-confusion-matrix-percent.png)

*Test confusion matrix: 10,947 / 1,553 / 1,404 / 11,096 — errors balanced across classes.*
