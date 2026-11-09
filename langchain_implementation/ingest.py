import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

MODEL = "gpt-6-luna"
DB_NAME = str(Path(__file__).parent.parent / "vector_db");
KNOWLEDGE_BASE_DIR = str(Path(__file__).parent.parent / "knowledge-base");
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize and constants

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")

MODEL = 'gpt-6-luna'
openai = OpenAI()

print(f"Database name: {DB_NAME}")

def fetch_files(directory):
    documents = []

    for folder in Path(directory).iterdir():
        if not folder.is_dir():
            continue

        loader = DirectoryLoader(
            str(folder),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = folder.name
            documents.append(doc)

    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    if (os.path.exists(DB_NAME)):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
        print(f"Deleted existing collection: {DB_NAME}")

    store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_NAME)

    return store


# if __name__ == "__main__":
#     documents = fetch_files(KNOWLEDGE_BASE_DIR)
#     chunks = create_chunks(documents)
#     store = create_embeddings(chunks)
#     print(f"Store: {store}")

if __name__ == "__main__":
    documents = fetch_files(KNOWLEDGE_BASE_DIR)
    print(f"Документов загружено: {len(documents)}")
    if not documents:
        raise SystemExit(f"Ничего не найдено в {KNOWLEDGE_BASE_DIR}")

    chunks = create_chunks(documents)
    print(f"Чанков создано: {len(chunks)}")
    store = create_embeddings(chunks)
    print(f"Store: {store}")