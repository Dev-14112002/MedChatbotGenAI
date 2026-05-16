import streamlit as st

# ---------------- PAGE ----------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

st.title("🩺 Medical AI Assistant")
from dotenv import load_dotenv
import os
import yaml
import streamlit_authenticator as stauth


from yaml.loader import SafeLoader

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory

from langchain.chains import create_retrieval_chain, create_history_aware_retriever

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks.base import BaseCallbackHandler

# Your project imports
from src.helper import download_embeddings
from src.prompt import qa_prompt, contextualize_q_prompt

# ---------------- LOAD ENV ----------------
load_dotenv()

with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authentication_status = authenticator.login()

name = st.session_state.get("name")
username = st.session_state.get("username")

if authentication_status:

    authenticator.logout("Logout", "sidebar")

    st.sidebar.write(f"Welcome {name}")

    st.title("🩺 Medical AI Assistant")

    # ALL YOUR CHATBOT CODE HERE

elif authentication_status is False:

    st.error("Incorrect username or password")

elif authentication_status is None:

    st.warning("Please login")


class StreamHandler(BaseCallbackHandler):

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


# ---------------- EMBEDDINGS ----------------
embeddings = download_embeddings()

# ---------------- VECTOR STORE ----------------
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-bot", embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ---------------- LLM ----------------
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)

# ---------------- MEMORY ----------------
if "memory" not in st.session_state:

    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

memory = st.session_state.memory

# ---------------- RAG ----------------
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------
prompt = st.chat_input("Ask your question")

if prompt:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Load memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Generate answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        stream_handler = StreamHandler(message_placeholder)

        response = rag_chain.invoke(
            {"input": prompt, "chat_history": chat_history},
            config={"callbacks": [stream_handler]},
        )

        answer = response["answer"]

        sources = response.get("context", [])

        if sources:
            with st.expander("📚 Sources"):
                for i, doc in enumerate(sources):

                    source = doc.metadata.get("source", "Unknown")

                    page = doc.metadata.get("page", "N/A")

                    st.markdown(f"### Source {i+1}")

                    st.write(f"**File:** {source}")
                    st.write(f"**Page:** {page}")

                    st.write(doc.page_content[:300] + "...")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save memory
    memory.save_context({"input": prompt}, {"output": answer})
