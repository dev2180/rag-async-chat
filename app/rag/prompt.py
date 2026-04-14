"""
MODULE: app/rag/prompt.py

Builds prompts using:
- Chat history
- Retrieved document context
"""

def build_prompt(query: str, context_chunks: list[str], history: list[dict]) -> str:

    context_text = "\n\n".join(context_chunks)

    history_text = ""
    for msg in history:
        history_text += f"{msg['role'].upper()}: {msg['content']}\n"

    return f"""
You are a helpful AI assistant answering questions strictly based on provided document context.

Instructions:
- Use ONLY the document context.
- Synthesize the answer in clear natural language.
- Do NOT copy sentences verbatim from the context.
- Summarize the relevant information.
- When possible, mention which source document supports your answer (e.g. "[filename.pdf]").
- You may infer conclusions logically from the provided context.
- Do NOT invent facts not supported by context.
- If not found, say exactly:
  "Not found in the provided PDFs. Sorry."


Chat History:
{history_text}

Document Context:
{context_text}

User Question:
{query}

Answer:
""".strip()
