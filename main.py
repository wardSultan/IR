import math
import codecs
import time

import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ir_datasets
import os
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import json
import numpy as np
from fuzzywuzzy import fuzz

import re
import enchant
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.tag import pos_tag
from sklearn.metrics.pairwise import cosine_similarity

# dataset = ir_datasets.load("beir/quora")
# dataset = ir_datasets.load("beir/fever")
# dataset = ir_datasets.load("beir/fever/test")
# dataset = ir_datasets.load("beir/fever/train")
dataset = ir_datasets.load("beir/nq")

# dataset = ir_datasets.load("beir/arguana")
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

folder_path = 'C:/Users/ward/Desktop/IR/corpus'
file_count = 0
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
relevant_documents_in_corpus=[]
corpus='C:/Users/ward/Desktop/IR/docs.json'

inverted_index_json='C:/Users/ward/Desktop/IR/inverted_index.json'

def preprocess_text(text):
    text.lower().replace('.', '').replace("'", '')
    words=word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    stopWords_filtered_words = []
   
    for w in words:
        #if w not in stop_words and  len(w) > 2 and not w.__contains__('.') and not w.__contains__("'"):
        if not w in stop_words and len(w) > 2: 
            stopWords_filtered_words.append(w)

    # remove puntion marks    
    filtered_words = [word for word in stopWords_filtered_words if word.isalnum()]
    
    stemmed_words = []
    ps = PorterStemmer()
    for w in filtered_words:
        stemmed_word = ps.stem(w)
        stemmed_words.append(stemmed_word)

    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()
    for w in stemmed_words:
        lemmatized_word = lemmatizer.lemmatize(w)
        lemmatized_words.append(lemmatized_word)
    
    unique_words = []
    for word in lemmatized_words:
        if word not in unique_words:
            unique_words.append(word)
    # Join the processed words back into a single string
    processed_text = ' '.join(unique_words)    
    return processed_text


# **************************************************************************************
docs = {}
doc_count=0
for doc   in dataset.docs_iter():
      if doc_count >= 100000:
             break
      docs[doc.doc_id] = preprocess_text(doc.text)
      doc_count+=1
  
with open(corpus, 'w') as json_file:
    json.dump(docs, json_file,indent=4)

# **************************************************************************************


def create_inverted_index(corpus):
    inverted_index = defaultdict(list)
    
    for doc_id, doc_content in corpus.items():
        terms = doc_content
        terms=terms.split()
        for term in terms:
            inverted_index[term].append(doc_id)
    return dict(inverted_index)



def process_query(query, inverted_index):
    query_terms=preprocess_text(query).split()
    print(query_terms)
    
    relevant_documents = set()

    for term in query_terms:
        if term in inverted_index:
            relevant_documents.update(inverted_index[term])

    return relevant_documents



def calculate_tfidf(relevant_documents):
    for document in relevant_documents:
      relevant_documents_in_corpus.append({document: json_data[document]})
    ward = {list(item.keys())[0]: list(item.values())[0] for item in relevant_documents_in_corpus}  
    documents = list(ward.values())

    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix

def rank_documents(query, tfidf_weights):
      # process query
    processed_query = preprocess_text(query)
    query_tfidf = vectorizer.transform([processed_query])
    result = cosine_similarity(tfidf_weights, query_tfidf).flatten()
     # Sort documents based on similarity scores
    ranked_indices = result.argsort()[::-1]

    # Return ranked documents
    ranked_documents = [(index, result[index]) for index in ranked_indices]

    return ranked_documents
# def rank_documents(query, tfidf_weights):
#     # Preprocess the query
#     query = preprocess_text(query)

#     # Calculate the TF-IDF weights for the query
#     query_weights = defaultdict(float)
#     for term in query.split():
#         query_weights[term] += 1

#     # Calculate the cosine similarity between the query and each document
#     similarity_scores = {}
#     for doc, weights in tfidf_weights.items():
#         dot_product = sum(query_weights[term] * weights.get(term, 0) for term in query_weights)
#         query_norm = math.sqrt(sum(query_weights[term] ** 2 for term in query_weights))
#         doc_norm = math.sqrt(sum(value ** 2 for value in weights.values()))
#         similarity_scores[doc] = dot_product / (query_norm * doc_norm)

#     # Sort the documents based on their similarity scores (descending order)
#     ranked_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

#     return ranked_documents
file_count=0 
tf_idf_docs=[]
my_list=[]
# Open the JSON file
with open(corpus, 'r') as file:
    # Load the JSON data
    json_data = json.load(file)
    



print("\n\n")
# corpus = {list(item.keys())[0]: list(item.values())[0] for item in my_list}

inverted_index = create_inverted_index(json_data)
with open(inverted_index_json, 'w') as json_file:
    json.dump(inverted_index, json_file,indent=4)
# Open the input JSON file
with open(inverted_index_json, 'r') as file:
    # Load the JSON data
    inverted_index = json.load(file)

# print("******************************"+"INVERTED INDEX"+"****************************************")
# print(inverted_index)
 
query_file_name="test6.txt"
queries_path = 'C:/Users/ward/Desktop/IR/queries'  
queries_file_path = os.path.join(queries_path, query_file_name)

if os.path.isfile(queries_file_path):
    with open(queries_file_path, 'r', encoding='utf-8') as file:
        queries_file_content = file.read()
        print('queries Name:', query_file_name)
        print('queries Content:', queries_file_content)
else:
    print('File not found:', query_file_name)
query=queries_file_content  
relevant_documents = process_query(query, inverted_index)
print("******************************"+"RELEVANT DOCUMENTS"+"****************************************")
print("Query:", query)

print(len(relevant_documents))

tfidf_weights = calculate_tfidf(relevant_documents)

# print("******************************"+"TF IDF MATRIX"+"****************************************")
# print(tfidf_weights)
# Rank the relevant documents
# vectorizer.fit(list(json_data.values()))
ranked_documents = rank_documents(query, tfidf_weights)
relevant_documents = list(relevant_documents)
print("Ranked Documents:")
for i, doc in enumerate(ranked_documents[:1000]):
    print(f"{i+1}. {doc}"+" , " +relevant_documents[i])
   

def calculate_precision_at_k(retrieved_documents, relevant_documents, k):
    retrieved_k = retrieved_documents[:k]
    relevant_and_retrieved = set(retrieved_k) & set(relevant_documents)
    precision = len(relevant_and_retrieved) / k
    return precision




qrels_path = 'C:/Users/ward/Desktop/IR/qrels'  



qrels_file_path = os.path.join(qrels_path, query_file_name)

if os.path.isfile(qrels_file_path):
    with open(qrels_file_path, 'r', encoding='utf-8') as file:
        qrels_file_content = file.read()
        print('qrels Name:', query_file_name)
        print('qrels Content:', qrels_file_content)
else:
    print('File not found:', query_file_name)




def recall(relevant_docs, retrieved_docs):
    num_relevant = len(set(relevant_docs).intersection(retrieved_docs))
    recall = num_relevant / len(relevant_docs)
    return recall

def average_precision(relevant_docs, retrieved_docs):
    precision_sum = 0.0
    num_relevant = 0
    
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            precision_sum += precision
    
    if num_relevant == 0:
        return 0.0
    
    average_precision = precision_sum / num_relevant
    return average_precision

def mean_average_precision(query_results):
    average_precisions = []
    
    for results in query_results:
        relevant_docs = results['relevant_docs']
        retrieved_docs = results['retrieved_docs']
        
        ap = average_precision(relevant_docs, retrieved_docs)
        average_precisions.append(ap)
    
    mean_ap = sum(average_precisions) / len(average_precisions)
    return mean_ap

def reciprocal_rank(relevant_docs, retrieved_docs):
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    
    return 0.0

def mean_reciprocal_rank(query_results):
    rr_sum = 0.0
    num_queries = len(query_results)
    
    for results in query_results:
        relevant_docs = results['relevant_docs']
        retrieved_docs = results['retrieved_docs']
        
        rr = reciprocal_rank(relevant_docs, retrieved_docs)
        rr_sum += rr
    
    mrr = rr_sum / num_queries
    return mrr

# Example usage
query_results = [
    {'relevant_docs': [1, 3, 5], 'retrieved_docs': [1, 2, 3, 4, 5]},
    {'relevant_docs': [2, 4, 6], 'retrieved_docs': [1, 2, 3, 4, 5, 6]},
    {'relevant_docs': [3, 5, 7], 'retrieved_docs': [1, 2, 3, 4, 5, 6, 7]},
]


retrieved_documents =str(qrels_file_content+".txt") 
query_results=[{'retrieved_docs':retrieved_documents,'relevant_docs': list(str(relevant_documents))}]
k = 10



precision_at_10 = calculate_precision_at_k(query_results[0]['relevant_docs'], query_results[0]['retrieved_docs'], k )
recall_score = recall(query_results[0]['relevant_docs'], query_results[0]['retrieved_docs'])
map_score = mean_average_precision(query_results)
mrr_score = mean_reciprocal_rank(query_results)

print(f"Precision@10: {precision_at_10}")
print(f"Recall: {recall_score}")
print(f"MAP: {map_score}")
print(f"MRR: {mrr_score}")
