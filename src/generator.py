def create_prompt(context, question):

    prompt = f"""
You are a helpful AI assistant.

Answer the question using only the provided context.

If the answer is not available in the context,
say "I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt



def generate_answer(llm, context, question):

    prompt = create_prompt(
        context,
        question
    )

    response = llm.invoke(prompt)

    return response