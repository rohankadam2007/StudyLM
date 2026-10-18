import os
from groq import Groq

# Initialize Groq client
# Reads from GROQ_API_KEY environment variable (locally or in Streamlit Cloud secrets)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You answer questions using ONLY the context given by the user.
Rules:
- Use only facts and examples that are written in the context.
- Do NOT invent tables, data, or examples.
- If the answer is not in the context, say exactly: "I don't know based on these notes."
- Keep the answer short."""

def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"

def ask_model(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content