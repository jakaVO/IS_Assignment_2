import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Specify the path to your JSON file
json_file_path = 'News_Category_Dataset_IS_course.json'

# Load only the first 1000 lines from the JSON file
with open(json_file_path, 'r') as file:
    data = [json.loads(line) for line in file.readlines()[:1000]]

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Drop rows with missing short_description
df = df.dropna(subset=['short_description'])

# Encode categories into numerical labels
df['category_label'] = pd.Categorical(df['category']).codes

# Select relevant columns
df = df[['short_description', 'category_label']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['short_description'], df['category_label'], test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
logistic_regression_predictions = logistic_regression_model.predict(X_test_tfidf)

# Evaluate the Logistic Regression model
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
print("Logistic Regression Accuracy:", logistic_regression_accuracy)
print(classification_report(y_test, logistic_regression_predictions))

# Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
random_forest_predictions = random_forest_model.predict(X_test_tfidf)

# Evaluate the Random Forest model
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
print("\nRandom Forest Accuracy:", random_forest_accuracy)
print(classification_report(y_test, random_forest_predictions))