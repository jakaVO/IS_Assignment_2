import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from gensim.models import Word2Vec

from bert_serving.client import BertClient


json_file_path = 'News_Category_Dataset_IS_course.json'

with open(json_file_path, 'r') as file:
    lines = file.readlines()

# Join the lines to form valid JSON
json_data = '[' + ','.join(lines) + ']'

# Read the JSON data into a Pandas DataFrame
df = pd.read_json(json_data)

# Extract the first 1000 rows and all columns
data = df.loc[:999, :]

print(data.head())

# Download necessary resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer and Stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    if text is None:
        return ''  # Return an empty string for missing values

    # Tokenize the text into words
    words = word_tokenize(text.lower())  # Convert text to lowercase

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words = [word.translate(table) for word in words if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Stemming (uncomment if you want to use stemming)
    stemmed_words = [stemmer.stem(word) for word in words]

    # Join the words back into a string
    preprocessed_text = ' '.join(lemmatized_words)
    return preprocessed_text

data['clean_text'] = data["short_description"].apply(preprocess_text)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['clean_text', 'short_description']], data['category'], test_size=0.2, random_state=42)
# Model building: Choose and train a classifier
vectorizer = TfidfVectorizer()  # Use TF-IDF vectorizer for text to numerical feature conversion
X_train_vec = vectorizer.fit_transform(X_train['clean_text'])
X_test_vec = vectorizer.transform(X_test['clean_text'])

# Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_vec, y_train)
logistic_predictions = logistic_model.predict(X_test_vec)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
print("Logistic Regression Accuracy:", logistic_accuracy)

# Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)
rf_predictions = rf_model.predict(X_test_vec)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)

# Tokenized text 
tokenized_train_text = [text.split() for text in X_train['clean_text']]
tokenized_test_text = [text.split() for text in X_test['clean_text']]

# Train Word2Vec model
w2v_model = Word2Vec(tokenized_train_text, vector_size=100, window=5, min_count=1, workers=4, epochs=10)

# Get all words and their vectors in the Word2Vec model's vocabulary
all_words = w2v_model.wv.index_to_key
word_vectors = {word: w2v_model.wv[word] for word in all_words[:5]}

# Print the word vectors
for word, vector in word_vectors.items():
    print(f"Word: {word}")
    print(f"Vector: {vector}")
    print("\n")  # Add a newline for better readability

# # Find similar words to a specific word
# similar_words = w2v_model.wv.most_similar('car', topn=10)

# 'word' is the word for which you want to find similar words, and 'topn' specifies the number of similar words to retrieve

# Print the similar words and their similarity scores
# for word, similarity in similar_words:
#     print(f"Similar word: {word}, Similarity: {similarity}")

# BERT

# split into training and validation sets
# split into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X_train['clean_text'], y_train, test_size=0.25, random_state=42)

# Comment out the following line as it's not applicable to Series
# print('X_tr shape:', X_tr.shape)

# make a connection with the BERT server using its IP address
bc = BertClient(ip="SERVER_IP")
# get the embedding for train and val sets
X_tr_bert = bc.encode(X_tr.tolist())
X_val_bert = bc.encode(X_val.tolist())

# LR model
model_bert = LogisticRegression()
# train
model_bert = model_bert.fit(X_tr_bert, y_tr)
# predict
pred_bert = model_bert.predict(X_val_bert)

print(accuracy_score(y_val, pred_bert))