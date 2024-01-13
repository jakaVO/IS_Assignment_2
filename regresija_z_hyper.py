import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Hyperparameter tuning for Logistic Regression
logistic_regression_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_grid = GridSearchCV(logistic_regression_model, logistic_regression_params, cv=5, scoring='accuracy')
logistic_regression_grid.fit(X_train_tfidf, y_train)

# Print the best hyperparameters for Logistic Regression
print("Best Hyperparameters for Logistic Regression:")
print(logistic_regression_grid.best_params_)

# Make predictions on the test set using the best Logistic Regression model
logistic_regression_predictions = logistic_regression_grid.predict(X_test_tfidf)

# Evaluate the Logistic Regression model
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
print("\nLogistic Regression Accuracy (with hyperparameter tuning):", logistic_regression_accuracy)
print(classification_report(y_test, logistic_regression_predictions))

# Hyperparameter tuning for Random Forest
random_forest_params = {'n_estimators': [50, 100, 150, 200], 'max_depth': [None, 10, 20, 30, 40]}

random_forest_model = RandomForestClassifier(random_state=42)
random_forest_grid = GridSearchCV(random_forest_model, random_forest_params, cv=5, scoring='accuracy')
random_forest_grid.fit(X_train_tfidf, y_train)

# Print the best hyperparameters for Random Forest
print("\nBest Hyperparameters for Random Forest:")
print(random_forest_grid.best_params_)

# Make predictions on the test set using the best Random Forest model
random_forest_predictions = random_forest_grid.predict(X_test_tfidf)

# Evaluate the Random Forest model
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
print("\nRandom Forest Accuracy (with hyperparameter tuning):", random_forest_accuracy)
print(classification_report(y_test, random_forest_predictions))
