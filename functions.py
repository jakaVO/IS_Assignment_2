import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import time

def scrape_website_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text() # Extract all text content from the HTML
            return text_content
        else:
            print(f"Failed to retrieve, status: {response.status_code}")
    except Exception as e:
        print(f"error : {e}")

def tfidf_vectorizer(text):
    # Pripravimo korpus, kjer je vsak dokument v korpusu enak vstopnemu besedilu
    corpus = [text]  # Pretvorimo vhodno besedilo v seznam dokumentov
    vectorizer = TfidfVectorizer() # Ustvarimo TF-IDF vektorizer
    tfidf_matrix = vectorizer.fit_transform(corpus) # Transformacija korpusa v matriko TF-IDF (Vsak stolpec v matriki predstavlja eno besedo, vsaka vrstica pa en dokument)
    tfidf_feature_names = vectorizer.get_feature_names_out() # Pridobimo besede iz vektorizerja
    tfidf_vector = tfidf_matrix.toarray()[0]  # Iz matrike izluscimo prvo vrstico (ker imamo samo en dokument)
    return dict(zip(tfidf_feature_names, tfidf_vector)) # Pretvorimo  v obliko slovarja (kljuci besede, vrednosti pa ustrezajo vrednostim matrike TF-IDF za posamezno besedo)

def word2vec_vectorizer(text):
    tokenized_text = word_tokenize(text.lower()) # Tokenizacija vhodnega besedila (razclenitev besedila v seznam besed, kjer so besede pretvorjene v male crke)
    # Model se uci na podlagi tokeniziranega besedila s parametri, kot so velikost vektorja (vector_size), okno (window), minimalno stevilo pojavitev besede (min_count), in stevilo delavcev (workers).
    model = Word2Vec(sentences=[tokenized_text], vector_size=10, window=5, min_count=1, workers=4) # Ustvarimo model Word2Vec
    word_vectors = {word: model.wv[word].tolist() for word in tokenized_text} # Pridobimo vektorje besed (Rezultat je slovar, kjer so kljuci besede, vrednosti pa vektorji besed.)
    return word_vectors


# Primer 
start_time = time.time()
website_url = "https://www.huffpost.com/entry/funniest-parenting-tweets_l_632d7d15e4b0d12b5403e479"
text = scrape_website_text(website_url)
if text:
    print(text)

end_time = time.time()
print("elapesed time: ", end_time-start_time)
"""
tfidf_result = tfidf_vectorizer(text)
print("TF-IDF Vector:", tfidf_result)
print()
word2vec_result = word2vec_vectorizer(text)
print("Word2Vec Embeddings:", word2vec_result)"""
