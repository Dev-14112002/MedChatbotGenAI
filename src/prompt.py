from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Given the chat history and the latest user question,
rewrite the question so it becomes standalone, clear,
and preserves the original meaning.
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful medical AI assistant.

Use the provided context to answer accurately.

IMPORTANT:
- Keep answers conversational.
- Use markdown formatting.
- Use headings when useful.
- Use bullet points for lists.
- Keep responses concise and readable.
- Highlight important terms using **bold** text.
- Avoid long textbook-style answers.

If the answer is not found in the context, say:
"I don't know based on the provided context."

Context:
{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
