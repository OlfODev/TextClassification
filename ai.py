import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import os

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv("train.csv")

# Prepare the data
X = data["comment_text"]
y = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Scale the data using StandardScaler
scaler = StandardScaler(with_mean=False)
X_train_tfidf = scaler.fit_transform(X_train_tfidf)
X_test_tfidf = scaler.transform(X_test_tfidf)

# Check if the trained model file exists
model_filename = "text_classification_model.joblib"
if not os.path.exists(model_filename):
    # Train a multi-label classification model (Logistic Regression) with a loading bar
    classifier = OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=1000))

    # Define the number of epochs (training iterations)
    epochs = 1

    # Initialize tqdm with the number of epochs and a narrower progress bar
    for epoch in tqdm(range(epochs), desc="Training Progress", ncols=80):
        classifier.fit(X_train_tfidf, y_train)

    # Save the trained model to a file
    joblib.dump(classifier, model_filename)
else:
    # Load the trained model from the file
    classifier = joblib.load(model_filename)

# Function to classify text and return label names with value 1
def classify_text(input_text):
    input_text_tfidf = tfidf_vectorizer.transform([input_text])
    input_text_tfidf = scaler.transform(input_text_tfidf)
    predictions = classifier.predict(input_text_tfidf)

    # Get the label names with value 1
    predicted_labels = [y.columns[i] for i, prediction in enumerate(predictions[0]) if prediction == 1]
    
    return predicted_labels

# Input text and get predicted label names with value 1
#while True:
    input_text = input("Enter text: ")
    results = classify_text(input_text)
    print("Predicted labels:", results)
