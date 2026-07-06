"""
Chat With My Resume — Flask backend
Pure-Python TF-IDF cosine similarity (no scikit-learn / numpy needed) to match
the user's question to the closest pre-written Q&A pair in knowledge_base.json.

Why pure Python? scikit-learn needs a compiled build for your exact Python
version. On brand-new Python versions (e.g. 3.14 on Windows) there's often no
pre-built wheel yet, so pip tries to compile from source and fails without a
C++ toolchain. This version avoids that entirely — just `pip install flask`.
"""

import json
import os
import re
import math
import random
from collections import Counter
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---- Load knowledge base ----
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

with open(KB_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

questions = [item["question"] for item in knowledge_base]
answers = [item["answer"] for item in knowledge_base]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "you", "your", "yours", "i", "me", "my", "what", "which", "who",
    "whom", "this", "that", "to", "of", "in", "on", "for", "and", "or",
    "it", "with", "as", "at", "by", "be", "have", "has", "had", "can",
    "tell", "about", "please", "thanks", "thank", "you're"
}

# ---- Greeting responses ----
GREETINGS = [
    "Hello! 👋 I'm Sadia's resume chatbot. Ask me about my projects, skills, education, or how to contact me!",
    "Hi there! 😊 I know all about Sadia — her projects, skills, education, and more. What would you like to know?",
    "Hey! 🤖 I'm here to answer questions about Sadia. Try asking about her projects, tech stack, or internship availability!",
    "👋 Welcome! I'm Sadia's AI assistant. Ask me anything about her background, skills, or projects!"
]

# ---- Fallback answer (more helpful) ----
FALLBACK_ANSWER = (
    "I don't have that information in my knowledge base yet. 📚\n\n"
    "Here's what I can tell you about:\n"
    "• Who Sadia is and her education (3.45 CGPA, 6th semester done)\n"
    "• Her projects (ResumeIQ, FraudGuard, Doctor App, and more)\n"
    "• Her skills (React, Flask, ML, Flutter, etc.)\n"
    "• Her summer break plans and internship availability\n"
    "• How to contact her\n\n"
    "Try one of the suggestions below! 👇"
)

CONFIDENCE_THRESHOLD = 0.15  # Increased from 0.12 for better accuracy


def tokenize(text: str):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


# ---- Build TF-IDF vectors for all stored questions ----
tokenized_questions = [tokenize(q) for q in questions]

# Document frequency: how many questions each word appears in
df = Counter()
for tokens in tokenized_questions:
    for word in set(tokens):
        df[word] += 1

N = len(tokenized_questions)


def tfidf_vector(tokens):
    tf = Counter(tokens)
    vector = {}
    for word, count in tf.items():
        idf = math.log((N + 1) / (df.get(word, 0) + 1)) + 1
        vector[word] = count * idf
    return vector


question_vectors = [tfidf_vector(tokens) for tokens in tokenized_questions]


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    common_words = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[w] * vec_b[w] for w in common_words)

    magnitude_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    magnitude_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def is_greeting(text: str) -> bool:
    """Check if user is just saying hello"""
    greetings = ["hi", "hello", "hey", "howdy", "greetings", "good morning", 
                 "good afternoon", "good evening", "sup", "yo", "hola", "namaste"]
    words = text.lower().split()
    return any(word in greetings for word in words) and len(words) <= 3


def log_unanswered(question: str):
    """Log unanswered questions to improve knowledge base"""
    try:
        log_file = os.path.join(os.path.dirname(__file__), "unanswered_questions.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            from datetime import datetime
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {question}\n")
    except:
        pass  # Silent fail - don't crash if logging fails


def get_best_answer(user_question: str) -> str:
    # Check if it's a greeting
    if is_greeting(user_question):
        return random.choice(GREETINGS)
    
    tokens = tokenize(user_question)
    if not tokens:
        return random.choice(GREETINGS)

    # Reuse the same idf weights computed from the stored questions
    user_vector = {}
    tf = Counter(tokens)
    for word, count in tf.items():
        idf = math.log((N + 1) / (df.get(word, 0) + 1)) + 1
        user_vector[word] = count * idf

    scores = [cosine_similarity(user_vector, qv) for qv in question_vectors]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]

    # If confidence is too low, log and return fallback
    if best_score < CONFIDENCE_THRESHOLD:
        log_unanswered(user_question)  # Track what users are asking
        return FALLBACK_ANSWER

    return answers[best_idx]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    reply = get_best_answer(user_message)
    return jsonify({"reply": reply})


@app.route("/api/suggestions", methods=["GET"])
def suggestions():
    # Most important questions that showcase Sadia best
    important = [
        "Who is Sadia? Tell me about yourself.",
        "What are you doing during your summer break?",
        "What projects have you worked on?",
        "What technologies or skills do you know?",
        "Are you looking for an internship?",
        "How can I contact you?"
    ]
    return jsonify({"suggestions": important})


if __name__ == "__main__":
    app.run(debug=True, port=5000)