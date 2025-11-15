import os
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GROQ CLIENT
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

documents = []
doc_names = []
vectorizer = None
tfidf_matrix = None


# -------------------------
# LOAD RAG DOCUMENTS
# -------------------------
def load_rag_documents():
    global documents, doc_names, vectorizer, tfidf_matrix

    documents = []
    doc_names = []

    print("Loading RAG data...")

    for file_path in glob.glob("rag/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            documents.append(text)
            doc_names.append(os.path.basename(file_path))

    if not documents:
        print("No documents found in rag folder!")
        return

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    print(f"Loaded {len(documents)} documents.")


@app.on_event("startup")
def startup_event():
    load_rag_documents()


# -------------------------
# LOCAL RAG SEARCH
# -------------------------
def rag_search(query):
    global vectorizer, tfidf_matrix, documents

    if vectorizer is None:
        return ""

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < 0.1:
        return ""

    return documents[best_index]


# -------------------------
# QUERY ENDPOINT
# -------------------------
@app.post("/query")
def query(q: str = Form(...)):
    context = rag_search(q)

    prompt = f"""
Use the context below to answer the user's question.

Context:
{context}

Question:
{q}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ NEW WORKING MODEL
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return {"answer": answer, "context_used": context}
