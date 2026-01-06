# %%
from top2vec import Top2Vec
import pandas as pd
from docx import Document
import re
import string
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


text_input = widgets.Text(
    value='',
    placeholder='Choose topic',
    description='Input:',
    disabled=False
)



# %%
def preprocess_text(text):
    text = re.sub(r'^[A-Za-z0-9]+[:]', '', text).strip()  
    text = re.sub(r'H0(?:-\d+)?', '', text)  
    text = re.sub(r"\bwasn't\b", "was not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoesn't\b", "does not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan't\b", "cannot", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdidn't\b", "did not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhasn't\b", "has not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdon't\b", "do not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcouldn't\b", "could not", text, flags=re.IGNORECASE)

    return text

doc = Document("H0 en.docx")
cleaned_lines = [preprocess_text(para.text) for para in doc.paragraphs if para.text.strip()]
full_text = " ".join(cleaned_lines)
texts_list = [full_text]  


# %%
def split_text_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def chunk_sentences(sentences, chunk_size):
    for i in range(0, len(sentences), chunk_size):
        yield ' '.join(sentences[i:i + chunk_size])

def split_texts_list(texts_list, chunk_size):
    result = []
    for text in texts_list:
        sentences = split_text_into_sentences(text)
        result.extend(chunk_sentences(sentences, chunk_size))
    return [chunk for chunk in result if chunk.strip()]

chunk_size = 6
split_texts = split_texts_list(texts_list, chunk_size=chunk_size)
split_texts = [doc for doc in split_texts if isinstance(doc, str) and doc.strip() != ""]


print(f"Created {len(split_texts)} chunks.")


# %%

with open("H0_split_chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(split_texts):
        f.write(f"[Chunk {i}]\n")
        f.write(chunk.strip() + "\n\n")



# %%
model = Top2Vec(split_texts,  min_count= 2, split_documents= False,
                    hdbscan_args={'min_cluster_size': 4, 'min_samples': 2, 'cluster_selection_method': 'leaf'},
                    umap_args={'n_neighbors': 8, 'min_dist': 0.0, 'metric': 'cosine'},

)


# %%
topic_sizes, topic_nums = model.get_topic_sizes()
print(topic_sizes)


# %%
topic_words, word_scores, topic_nums = model.get_topics()

for i, word_scores, num in zip(topic_words, word_scores, topic_nums):
    print(num)
    print(f'Words: {i}')
from collections import defaultdict

# %%
client = OpenAI(api_key = "---insert-your-key-here----")
def label_topic_with_gpt(topic_keywords, topic_documents
                         ):
    prompt = (
    "You are an AI that labels discussion topics, from a cancer storytelling interview, "
    "for a software that allows doctors to browse through medical files without the need to read them from start "
    "to finish. Given the following keywords and sample documents, provide a clear and specific "
    "topic label, with enough context to be interpretable and to bring attention to the topic at hand, focusing mainly on the keyword list and using the document snippets as supporting"
    "context rather than a baseline. Max 15 words. You can hit the max. Only type the topic label and nothing else:\n\n"
    f"Keywords: {', '.join(topic_keywords)}\n\n"
    f"Sample Documents:\n{topic_documents[:5]}\n\n"
    "Label:"
)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that helps label topics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            temperature=0.5
        )
        
        topic_label = response.choices[0].message.content.strip()
        return topic_label
    
    except Exception as e:
        print(f"Error in GPT API request: {e}")
        return None

topic_labels = []
for i, topic_keywords in enumerate(topic_words):
    total_documents_for_topic = model.topic_sizes[i]
    num_docs = min(total_documents_for_topic, 10000)
    documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=i, num_docs = num_docs)
    
    label = label_topic_with_gpt(topic_keywords, documents
    )
    topic_labels.append((i, label))

for topic_num, label in topic_labels:
    print(f"Topic {topic_num}: {label}")

# %%

button = widgets.Button(description="Submit Topic Number")

display(text_input, button)

def on_button_click(b):
    from IPython.display import display
    display(f"Topic number entered: {text_input.value}")
    try:
        topic_num = int(text_input.value)  
        topic_label = topic_labels[topic_num][1]
        total_documents_for_topic = model.topic_sizes[topic_num]
        num_docs = min(total_documents_for_topic, 10000)
        documents, document_scores, document_ids = model.search_documents_by_topic(topic_num, num_docs = num_docs)
        display(f"Topic number {text_input.value} has label: {topic_label}")
        for doc, score, doc_id in zip(documents, document_scores, document_ids):
            display(f"Document: {doc_id}, Score: {score}")
            display(doc)
            display()
    except ValueError:
        display("Please enter a valid topic number.")
    display(text_input, button)


button.on_click(on_button_click)


# %%
