# === **1. Import Libraries**

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === **2. Load Dataset (Data Collection)**

train = pd.read_excel("IMDB_train.xlsx")
test = pd.read_excel("IMDB_test.xlsx")

train.head()

test.head()

train['review']

test['review']

# === **3. Exploratary Data Analysis (EDA) - Understand the data**

print("\n===== Basic Info =====")
print(train.info())

print("\n===== Missing Values =====")
print(train.isnull().sum())

# === **3.1 Class Distribution**

plt.figure()
train['sentiment'].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.show()

print(train['sentiment'].value_counts(normalize=True))

# === **3.2 Text Length**

train['text_length'] = train['review'].astype(str).apply(len)

plt.figure()
plt.hist(train['text_length'])
plt.title("Text Length Distribution")
plt.show()

plt.figure()
sns.boxplot(x=train['sentiment'], y=train['text_length'])
plt.title("Text Length by Class")
plt.show()

# === **3.3 Word Count**

train['word_count'] = train['review'].apply(lambda x: len(str(x).split()))

plt.figure()
plt.hist(train['word_count'])
plt.title("Word Count Distribution")
plt.show()

# === **3.4 Common Words**

from collections import Counter

all_words = " ".join(train['review'].astype(str)).lower().split()
common_words = Counter(all_words).most_common(20)

words_df = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure()
plt.bar(words_df['word'], words_df['count'])
plt.xticks(rotation=45)
plt.title("Top 20 Words")
plt.show()

# === **3.5 Word Cloud**

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400).generate(" ".join(train['review'].astype(str)))

plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud Before Cleaning")
plt.show()

# === **4. Pre-Processing**

custom_stopwords = set(['would','could','also','one','get',])
all_stopwords = stop_words.union(custom_stopwords)

def handle_negation(text):
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    return text

def clean_text(text):
    # Ensure the input is a string before applying string methods
    text = str(text).lower()
    text = handle_negation(text)

    text = re.sub(r'[^ɐ-\uffff\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in all_stopwords and len(w) > 2
    ]

    return " ".join(words)

train['clean_text'] = train['review'].apply(clean_text)
test['clean_text'] = test['review'].apply(clean_text)

plt.figure()
wordcloud = WordCloud(width=800, height=400).generate(" ".join(train['clean_text']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud After Cleaning")
plt.show()

common_words = Counter(" ".join(train['clean_text']).split()).most_common(20)

words_df = pd.DataFrame(common_words, columns=['word','count'])

plt.figure()
plt.bar(words_df['word'], words_df['count'])
plt.xticks(rotation=45)
plt.title("Top Words After Cleaning")
plt.show()

from collections import Counter

# split by sentiment
positive_text = " ".join(train[train['sentiment'] == 1]['clean_text'])
negative_text = " ".join(train[train['sentiment'] == 0]['clean_text'])

# count words
pos_counts = Counter(positive_text.split()).most_common(10)
neg_counts = Counter(negative_text.split()).most_common(10)

# convert to dataframe
import pandas as pd
pos_df = pd.DataFrame(pos_counts, columns=['word','count'])
neg_df = pd.DataFrame(neg_counts, columns=['word','count'])


# Side by side plot
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Positive
axes[0].barh(pos_df['word'], pos_df['count'])
axes[0].set_title("Top Words - Positive")
axes[0].invert_yaxis()

# Negative
axes[1].barh(neg_df['word'], neg_df['count'])
axes[1].set_title("Top Words - Negative")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# === **5. Feature Engineering (TF-IDF + NGram)**

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.8
)

X = vectorizer.fit_transform(train['clean_text'])
y = train['sentiment']

X_test = vectorizer.transform(test['clean_text'])
y_test = test['sentiment']

from sklearn.decomposition import PCA

# reduce TF-IDF to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

plt.figure()

# color by label
for label in np.unique(y):
    plt.scatter(
        X_reduced[y == label, 0],
        X_reduced[y == label, 1],
        label=f"Class {label}",
        alpha=0.6
    )

plt.legend()
plt.title("Text Distribution (PCA)")
plt.show()

# === **6. Train Validation Split**

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === **7. Model 1 - Logistic Regression**

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_val)
print("LR Accuracy:", accuracy_score(y_val, y_pred_lr))

print("\nLR Classification Report:\n")
print(classification_report(y_val, y_pred_lr))

print("LR Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lr))

# === **8. Model 2 - Naive Bayes**

nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_val)
print("NB Accuracy:", accuracy_score(y_val, y_pred_nb))

print("\nNB Classification Report:\n")
print(classification_report(y_val, y_pred_nb))

print("NB Confusion Matrix:\n", confusion_matrix(y_val, y_pred_nb))

# === **9. Model 3 - SVM**

svm = LinearSVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_val)
print("SVM Accuracy:", accuracy_score(y_val, y_pred_svm))

print("\nSVM Classification Report:\n")
print(classification_report(y_val, y_pred_svm))

print("SVM Confusion Matrix:\n", confusion_matrix(y_val, y_pred_svm))

# === **10. Model Comparison**

print("\nModel Comparison")
print("LR:", accuracy_score(y_val, y_pred_lr))
print("NB:", accuracy_score(y_val, y_pred_nb))
print("SVM:", accuracy_score(y_val, y_pred_svm))

print("LR F1:", classification_report(y_val, y_pred_lr, output_dict=True)['weighted avg']['f1-score'])
print("NB F1:", classification_report(y_val, y_pred_nb, output_dict=True)['weighted avg']['f1-score'])
print("SVM F1:", classification_report(y_val, y_pred_svm, output_dict=True)['weighted avg']['f1-score'])

# === **11. Hyperparameter Tuning**

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {'C': [0.1, 1, 10]}
    ),
    "Naive Bayes": (
        MultinomialNB(),
        {'alpha': [0.1, 1, 10]}
    ),
    "SVM": (
        LinearSVC(),
        {'C': [0.1, 1, 10]}
    )
}

# === **12. Cross Validation**

results = {}

for name, (model, params) in models.items():
    print(f"\n===== {name} =====")

    grid = GridSearchCV(model, params, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("Best Params:", grid.best_params_)

    # Cross validation (fast)
    cv_scores = cross_val_score(best_model, X, y, cv=3)
    cv_f1 = cross_val_score(best_model, X, y, cv=3, scoring='f1_weighted')

    print("CV Accuracy:", cv_scores.mean())
    print("CV F1 Score:", cv_f1.mean())

    # Test score
    test_score = best_model.score(X_test, y_test)
    print("Test Accuracy:", test_score)

    results[name] = {
        "model": best_model,
        "test_score": test_score
    }

best_model_name = max(results, key=lambda x: results[x]['test_score'])
best_model = results[best_model_name]['model']

print("\nBest Model:", best_model_name)

# === **13. Final test Validation**

final_pred = best_model.predict(X_test)

# Metrics
print("\nFinal Accuracy:", accuracy_score(y_test, final_pred))
print("\nClassification Report:\n", classification_report(y_test, final_pred))

cm = confusion_matrix(y_test, final_pred)
print("\nConfusion Matrix:\n", cm)

# Heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# Percentage
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure()
sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix (%)")
plt.show()

# === **14. Error Analysis**

test['predicted'] = final_pred

wrong = test[test['sentiment'] != test['predicted']]
print("\nWrong Predictions:")
print(wrong[['review', 'sentiment', 'predicted']].head(20))

# === **15. Custom Prediction Function**

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return best_model.predict(vec)[0]

print("\nCustom Movie Review Sentiment Testing:\n")

print("\nPositive Reviews:\n")
print(predict_sentiment("This movie was absolutely amazing, the storyline was engaging and the acting was brilliant"))
print(predict_sentiment("I really loved this film, the cinematography was beautiful and the plot was very touching"))
print(predict_sentiment("One of the best movies I have ever watched, highly recommended to everyone"))

print("\nNegative Reviews:\n")
print(predict_sentiment("This movie was terrible, the plot made no sense and the acting was very poor"))
print(predict_sentiment("I regret watching this film, it was boring and a complete waste of time"))
print(predict_sentiment("The movie was disappointing, nothing interesting happened and it was too slow"))

print("\nNegation Reviews:\n")
print(predict_sentiment("This movie is not good at all, I did not enjoy it"))
print(predict_sentiment("The film is not bad, actually quite entertaining in some parts"))
print(predict_sentiment("I am not impressed with this movie, it could have been much better"))

print("\nMixed Reviews:\n")
print(predict_sentiment("The visuals were stunning but the story was very weak"))
print(predict_sentiment("Great acting but the plot was confusing and poorly executed"))
print(predict_sentiment("I liked some parts of the movie but overall it was disappointing"))

print("\nNeutral Reviews:\n")
print(predict_sentiment("The movie was okay, nothing special but not too bad either"))
print(predict_sentiment("It was an average film, some good scenes but also some boring moments"))