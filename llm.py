import os
import streamlit as st
from groq import Groq

def get_groq_client():
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
        if env_key:
            api_key = env_key.strip()
        
    if not api_key:
        raise ValueError("GROQ_API_KEY missing in Secrets/Environment.")
        
    return Groq(api_key=api_key)

SYSTEM_PROMPT = """You answer questions using ONLY the context given by the user.
Rules:
- Use only facts and examples that are written in the context.
- Do NOT invent tables, data, or examples.
- If the answer is not in the context, say exactly: "I don't know based on these notes."
- Structure answers clearly with bullet points and bold headings."""

def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"

def ask_model(prompt):
    client = get_groq_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content