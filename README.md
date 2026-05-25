 Email Spam Detection using Machine Learning
 Project Overview

This project implements a Machine Learning based Email Spam Detection system using Python and Scikit-learn. The model analyzes email messages and classifies them as Spam or Not Spam.

The system uses TF-IDF Vectorization for text processing and the Multinomial Naive Bayes algorithm for classification.

This project demonstrates how Natural Language Processing (NLP) techniques can be applied to detect unwanted or malicious emails automatically.

-------------------Dataset-------------------

Dataset Link:
https://drive.google.com/file/d/1zNYO05u5dePV8rzTjevH1mFA9yDfDJWo/view

The dataset contains email messages used to train and test the spam detection model.

Dataset Features
Column	Description
message	The email text content
label	Spam (1) or Not Spam (0)

If labels are not available, the project automatically generates them using spam keyword detection.

-------------------Technologies Used-------------------

Python

Pandas

Scikit-learn

TF-IDF Vectorization

Multinomial Naive Bayes

-------------------Required Libraries-------------------

Install dependencies before running the project.

pip install pandas scikit-learn
Machine Learning Workflow

The project follows these steps:

1. Load Dataset

The dataset is loaded using Pandas.

df = pd.read_csv("emails.csv")

2️. Data Preprocessing

Convert column names to lowercase

Handle missing values

Extract message text

df.rename(columns=lambda x: x.lower(), inplace=True)
df["message"] = df["message"].fillna("")

3️. Spam Label Creation

Spam emails are detected using common spam keywords such as:

free

win

offer

urgent

money

lottery

claim

click

bonus

df["label"] = df["message"].str.lower().apply(
    lambda s: 1 if any(word in s for word in spam_keywords) else 0
)

4. Train Test Split

Dataset is divided into training (80%) and testing (20%).

train_test_split(X, y, test_size=0.2, random_state=42)

5. TF-IDF Text Vectorization

Text is converted into numerical form using TF-IDF.

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

6. Model Training

The project uses Multinomial Naive Bayes, a popular algorithm for text classification.

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

7️. Model Evaluation

Model performance is evaluated using:

Accuracy

Confusion Matrix

Classification Report

accuracy_score
confusion_matrix
classification_report

8️. Custom Email Testing

The system can test new messages and predict whether they are spam.

Example:

Message: Congratulations! You have won a $1000 Amazon gift card.
Prediction: Spam
Message: Let's meet after the CN Lab tomorrow.
Prediction: Not Spam

-------------------Example Output-------------------
Loaded 200 emails

Label distribution:
0    150
1     50

Model Evaluation:
Accuracy: 92.50 %

Confusion Matrix:
[[28  2]
 [ 1  9]]

Spam Filter Ready!

-------------------Project Structure-------------------
Email-Spam-Detection
│
├── emails.csv
├── spam_detector.py
├── README.md

-------------------Future Improvements-------------------

Possible improvements for the project:

Use real labeled spam datasets

Build a Flask Web App

Add Deep Learning (LSTM / BERT)

Deploy as an API

Integrate with email clients


-------------------Author-------------------

Aryan Gore
B.Tech CSE (III Year)
Shree Vaishnav Vidyapeeth Vishwavidyalaya