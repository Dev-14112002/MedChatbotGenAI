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
You are a helpful medical assistant.

Use the provided context and conversation history to answer the user's question accurately.

If the question is a follow-up question, interpret it using the previous conversation context.

IMPORTANT FORMATTING RULES:
- Use markdown formatting.
- Use headings when useful.
- Use bullet points for explanations or lists.
- Keep answers clean, readable, and well-structured.
- Use short paragraphs.
- Highlight important medical terms using **bold** text.
- Add spacing between sections.

If the answer is not present in the context, say:
"I don't know based on the provided context."

Context:
{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
