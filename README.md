# Chat With My Resume 🤖

A simple AI chatbot that answers questions about you using TF-IDF cosine similarity
(the same technique used in ResumeIQ) — no external LLM API needed, fully self-contained.

## How it works
1. Your info lives in `knowledge_base.json` as Q&A pairs.
2. When a user types a question, the backend converts it into a TF-IDF vector and
   compares it against all stored questions using cosine similarity.
3. The closest matching answer is returned. If nothing matches well enough, a
   friendly fallback message is shown instead.

## Setup (2 minutes)

```bash
cd chat-with-resume
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Customize it with YOUR info

Open `knowledge_base.json` and edit/add Q&A pairs. Format:

```json
{
  "question": "What projects have you built?",
  "answer": "I built X using Y, and Z using W..."
}
```

Tips:
- Add 8–15 Q&A pairs covering: who you are, top projects, skills, availability,
  contact info, and "why should I hire you" type questions.
- Write questions the way a recruiter might actually phrase them — the bot
  matches on wording similarity, so natural phrasing works best.
- No restart needed for JSON edits if you're using Flask's debug/reload mode
  (already enabled in `app.py`).

## Project structure
```
chat-with-resume/
├── app.py                 # Flask backend + matching logic
├── knowledge_base.json    # Your editable Q&A data
├── requirements.txt
├── templates/
│   └── index.html         # Chat UI (Tailwind, vanilla JS)
└── README.md
```
