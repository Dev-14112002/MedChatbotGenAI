from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_embeddings
from src.prompt import qa_prompt, contextualize_q_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain


# -------------------- APP INIT --------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("API keys are missing in environment variables")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -------------------- VECTOR STORE --------------------
embeddings = download_embeddings()

index_name = "medical-bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# -------------------- LLM --------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


# -------------------- MEMORY --------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# -------------------- RETRIEVER (HISTORY AWARE) --------------------
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


# -------------------- QA CHAIN --------------------
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# -------------------- FINAL RAG CHAIN --------------------
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    # Load chat history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Invoke RAG chain
    response = rag_chain.invoke({"input": msg, "chat_history": chat_history})

    # Save conversation back to memory
    memory.save_context({"input": msg}, {"output": response["answer"]})

    return response["answer"]


# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
