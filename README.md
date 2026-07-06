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

## Make it "impressive" for LinkedIn

- Record a 20–30 second screen recording of you asking it 3–4 questions.
- Post it with a caption like: *"Recruiters usually skim resumes in 6 seconds —
  so I built a chatbot that lets them just ask instead. Try it 👇"*
- Deploy it for free on **Render** or **Railway** so you can share a live link,
  not just a video (both support free Flask hosting).

## Optional upgrades (once the MVP is live)
- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`) for smarter matching.
- Add a "Download my resume" button in the chat header.
- Log unanswered questions to a file so you know what to add to the knowledge base.
- Connect it to an actual LLM API (OpenAI/Anthropic) for more natural, generative answers
  instead of fixed responses — swap `get_best_answer()` in `app.py` to call the API,
  using the matched Q&A as context.

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
