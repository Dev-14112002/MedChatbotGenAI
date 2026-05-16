from flask import Flask, render_template, request, redirect, jsonify, session
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
load_dotenv()
app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=["20 per minute"])
app.secret_key = os.getenv("SECRET_KEY")


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


# -------------------- Login Required Decorator -------------------


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect("/")
        return f(*args, **kwargs)

    return decorated_function


# -------------------- ROUTES --------------------
@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/chat")
@login_required
def index():
    return render_template("chat.html")


@app.route("/login", methods=["POST"])
def login():

    username = request.form.get("username")
    password = request.form.get("password")

    if username == "admin" and password == "medchat123":

        session["user"] = username

        return redirect("/chat")

    return "Invalid credentials"


@app.route("/get", methods=["POST"])
@login_required
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


@app.route("/logout")
@login_required
def logout():

    session.pop("user", None)

    return redirect("/")


# -------------------- RUN --------------------
"""if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
"""
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
