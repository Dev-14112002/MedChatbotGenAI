from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, "
            "rewrite the question so it is standalone and clear.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful medical assistant.\n"
            "Use the provided context to answer.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
