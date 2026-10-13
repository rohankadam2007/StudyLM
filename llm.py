def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on these notes."

Context:
{context}

Question: {question}

Answer:"""
    return prompt