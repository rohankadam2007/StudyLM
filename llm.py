import ollama
def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on these notes."

Context:
{context}

Question: {question}

Answer:"""
    return prompt

def ask_model(prompt):
    response = ollama.chat(
        model="llama3.2:3b",   # use the exact name from ollama list
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]