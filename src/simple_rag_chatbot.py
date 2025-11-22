import os
from getpass import getpass
from pathlib import Path
from groq import Groq
import chainlit as cl

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Ask for Groq key if env var not set
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or getpass(
    "Enter your Groq API key: "
)

SYSTEM_PROMPT = """You are a virtual assistant for the Central Bank of India.
Your goal is to assist users in using the bank app.
You must use the relevant documentation given to you to answer user queries.
You can only answer questions about the bank app. Incase you can not answer something reply with can't answer this question instead.
"""

# Build VectorStore Retriever
loader = PyPDFLoader(Path(__file__).parent / "CBIPPD.pdf")
pages = loader.load_and_split()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(pages)
print("Indexing documents...")
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
print("done!")
retriever = vector.as_retriever()


@cl.on_chat_start
async def on_chat_start():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    cl.user_session.set("client", client)

    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("chat_history", chat_history)

    cl.user_session.set("retriever", retriever)

    welcome_message = (
        "Welcome to the Central Bank of India App support center. "
        "How may I assist you?"
    )
    chat_history.append({"role": "assistant", "content": welcome_message})

    await cl.Message(content=welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    client: Groq = cl.user_session.get("client")
    chat_history: list = cl.user_session.get("chat_history")
    retriever: VectorStoreRetriever = cl.user_session.get("retriever")

    try:
        chat_history.append({"role": "user", "content": message.content})

        relevant_docs = retriever.get_relevant_documents(message.content)
        if relevant_docs:
            relevant_info = "\n\n".join([d.page_content for d in relevant_docs])
            chat_history.append(
                {"role": "system", "content": f"Relevant Documents:\n{relevant_info}"}
            )

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=chat_history,
        )
        reply = response.choices[0].message["content"]

        chat_history.append({"role": "assistant", "content": reply})
        cl.user_session.set("chat_history", chat_history)

        await cl.Message(content=reply).send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()


@cl.on_stop
async def on_stop():
    cl.user_session.set("chat_history", [])
