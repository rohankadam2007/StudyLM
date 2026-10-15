import ollama

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
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]