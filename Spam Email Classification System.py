import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text
df = pd.read_csv("spam.csv", encoding="latin1")
print("Columns in your dataset:\n", df.columns)
label_candidates = ['label', 'Label', 'Category', 'class', 'v1', 'target']
text_candidates = ['text', 'message', 'Message', 'email', 'Email', 'v2']
label_col = None
text_col = None
for col in df.columns:
    if col in label_candidates:
        label_col = col
    if col in text_candidates:
        text_col = col
if label_col is None or text_col is None:
    print("\nERROR: Could not detect label/text columns automatically.")
    print("Please tell me your dataset columns so I can fix it.")
    exit()
df = df[[label_col, text_col]]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1, 'Ham': 0, 'Spam': 1})
df = df.dropna()
df['text'] = df['text'].astype(str).apply(clean_text)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nReport:\n", classification_report(y_test, pred))
def predict_email(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    if prob[1] > 0.30:
        return "SPAM"
    else:
        return "HAM (Not Spam)"
while True:
    msg = input("\nEnter email (type 'exit'): ")
    if msg.lower() == "exit":
        break
    print("Prediction:", predict_email(msg))
