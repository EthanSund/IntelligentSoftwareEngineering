#dependencies
import pandas as pd
import numpy as np
import re
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from scipy import stats

#methods used for the baseline solution originate from, as referenced in the deliverable
#https://github.com/ideas-labo/ise-lab-solution/tree/main/lab1

#stopwords required
nltk.download('stopwords', quiet=True)
#model for semantic embeddings
nlp = spacy.load('en_core_web_md')

#-----BASELINE PREPROCESSING------
#cleans the text, strips formatting, punctuation, non-alphanumeric chars etc

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F" 
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

stop_words_list = stopwords.words('english')

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words_list])

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

#-----SABER (my solution)-----

class SpacyVectorizer(BaseEstimator, TransformerMixin):
    #Branch 1: Semantic Embeddings - converts natural lang into vector embeddings
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([nlp(str(text)).vector for text in X])

class HeuristicExtractor(BaseEstimator, TransformerMixin):
    #Brnach 2: Heuristic Extraction - uses RegEx on the description of the bug
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features=[]
        for text in X:
            t = str(text)
            #flag for errors
            exception = 1 if re.search(r'(Exception|Error|java\.lang)', t, re.IGNORECASE) else 0
            #count variables in CamelCase
            camel_case = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', t))
            description_length = len(t)
            features.append([exception, camel_case, description_length])
        return np.array(features)
#preprocessing - routes title to branch 1, body to branch 2
saber_preprocessor = ColumnTransformer(
    transformers=[
        ('titleSemantics', SpacyVectorizer(), 'Title'),
        ('descriptionHeuristics', HeuristicExtractor(), 'Body'),
    ]
)
#combination of the extraction, scaling needed for the SVM, and the SVM
saber_pipeline = Pipeline([
    ('preprocessor', saber_preprocessor),
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='linear', random_state=42))
])

#preparation

data = pd.read_csv('pytorch.csv').fillna('')
#removes rare classes (<5 instances) to prevent crashes during validation
valid_classes = data['class'].value_counts()[data['class'].value_counts() >= 5].index
data = data[data['class'].isin(valid_classes)]
#joins the title and body for the baseline technique
data['Title+Body'] = data.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if row['Body'] != '' else row['Title'], axis=1
)
#applies the baseline technique of preprocessing
text_col = 'text'
data[text_col] = data['Title+Body'].apply(remove_html).apply(remove_emoji).apply(remove_stopwords).apply(clean_str)

#10-fold evaluation
REPEAT=10
base_f1s, saber_f1s = [],[]
base_acc, saber_acc = [],[]

#hyperparameters for Naive Bayes (baseline)
params = {'var_smoothing': np.logspace(-12,0,13)}

for x in range(REPEAT):
    #train/test split 80/20.
    #random_state=x means identical splits for baseline & saber
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=x
    )
    #targets
    y_train = data['class'].iloc[train_index]
    y_test = data['class'].iloc[test_index]

    #BASELINE: TF-IDF + NB
    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000
    )
    X_train = tfidf.fit_transform(train_text).toarray()
    X_test = tfidf.transform(test_text).toarray()

    clf = GaussianNB()
    grid = GridSearchCV(
        clf,
        params,
        cv=5,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    best_clf.fit(X_train, y_train)
    base_pred = best_clf.predict(X_test)
    base_f1s.append(f1_score(y_test, base_pred, average='macro'))
    base_acc.append(accuracy_score(y_test, base_pred))

    #SABER
    #columns fed into saber pipeline
    X2_train = data[['Title','Body']].iloc[train_index]
    X2_test = data[['Title','Body']].iloc[test_index]

    saber_pipeline.fit(X2_train, y_train)
    saber_pred = saber_pipeline.predict(X2_test)

    saber_f1s.append(f1_score(y_test, saber_pred, average='macro'))
    saber_acc.append(accuracy_score(y_test, saber_pred))

    print(f"Loop {x + 1} of 10 Complete")

#results and analysis

#Wilcoxon Signed-Rank Test for metrics
stat, pVal = stats.wilcoxon(base_f1s, saber_f1s)
#results
print("\n")
print("FINAL MACRO F1-SCORES")
print(f"Baseline (TF-IDF + NB): {np.mean(base_f1s):.4f}")
print(f"SABER: {np.mean(saber_f1s):.4f}")
print(f"Baseline Accuracy: {np.mean(base_acc):.4f}")
print(f"SABER Accuracy: {np.mean(saber_acc):.4f}")
print(f"Wilcoxon P-value: {pVal:.5f}")

if np.mean(saber_f1s) > np.mean(base_f1s):
    print("SUCCESS")
else:
    print("FAIL")