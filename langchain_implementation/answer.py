from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "gpt-6-luna"
DB_NAME = str(Path(__file__).parent.parent / "vector_db");

KNOWLEDGE_BASE_DIR = str(Path(__file__).parent.parent / "knowledge-base");

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the user's documents.
You are given a question and a context.
You need to answer the question based on the context.
Context:
{context}
"""

vectorestore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorestore.as_retriever()

llm = ChatOpenAI(model=MODEL, temperature=0, reasoning_effort="none")

def answer_question(question, history=None) -> tuple[str, list[Document]]:
    history = history or []

    messages = convert_to_messages(history)       # принимает и str, и список блоков
    messages.append(HumanMessage(content=question))

    # весь текст пользователя для поиска, без ручных join
    user_text = "\n".join(m.text for m in messages if isinstance(m, HumanMessage))

    docs = retriever.invoke(user_text)
    context = "\n\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT.format(context=context)), *messages]
    )
    return response.text, docs