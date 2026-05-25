import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# STEP 1: Load CSV
# -----------------------------

csv_path = r"emails.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} emails")
print("Columns:", df.columns)

# Ensure message column exists
df.rename(columns=lambda x: x.lower(), inplace=True)
df["message"] = df["message"].fillna("")

# -----------------------------
# STEP 2: Simple spam labeling
# -----------------------------
spam_keywords = ["free", "win", "offer", "urgent", "money", "claim", "lottery", "click", "bonus"]
df["label"] = df["message"].str.lower().apply(lambda s: 1 if any(word in s for word in spam_keywords) else 0)

print("\nLabel distribution:")
print(df["label"].value_counts())

# -----------------------------
# STEP 3: Prepare text for training
# -----------------------------
X = df["message"]
y = df["label"]

# -----------------------------
# STEP 4: Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# STEP 5: TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# STEP 6: Train Naive Bayes model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# -----------------------------
# STEP 7: Evaluate model
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# STEP 8: Test custom messages
# -----------------------------
print("\nTesting custom messages:")
samples = [
    "Congratulations! You have won a $1000 Amazon gift card. Click here to claim your prize.",
    "Dear Sir, please find attached the report for tomorrow's DBMS presentation.",
    "URGENT: Your account will be deactivated. Verify immediately to avoid loss.",
    "Let's meet after the CN Lab tomorrow."
]

sample_tfidf = vectorizer.transform(samples)
predictions = model.predict(sample_tfidf)

for text, pred in zip(samples, predictions):
    print(f"\nMessage: {text}\nPrediction: {'Spam' if pred==1 else 'Not Spam'}")

print("\n Spam Filter Ready!")
