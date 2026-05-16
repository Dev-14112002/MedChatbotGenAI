import token

from flask import Flask, render_template, request, jsonify
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
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
limiter = Limiter(get_remote_address, app=app, default_limits=["20 per minute"])
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
APP_API_TOKEN = os.environ.get("APP_API_TOKEN")

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


# -------------------- AUTH DECORATOR -------------------
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Missing authorization token"}), 401

        if token != f"Bearer {APP_API_TOKEN}":
            return jsonify({"error": "Invalid token"}), 403
        return f(*args, **kwargs)

    return decorated


# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
@require_auth
@limiter.limit("10 per minute")
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
"""if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
"""
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
