# main.py
import os
import hashlib
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

client = Groq()
PERSIST_ROOT = "chroma_cache"  # Persistent cache dir (do NOT delete automatically)
os.makedirs(PERSIST_ROOT, exist_ok=True)

# Embedding model
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

# Helpers
def url_to_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def load_pdf_text(pdf_path):
    """Extract text from PDF using pdfplumber."""
    print(f"📄 Loading PDF: {pdf_path}")
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if not page_text:
                continue
            text_chunks.append(page_text)
            if (i + 1) % 10 == 0:
                print(f"✅ Processed {i + 1}/{len(pdf.pages)} pages")
    return "\n\n".join(text_chunks)

def split_text(text, chunk_size=500, chunk_overlap=100):
    """Split large text into overlapping chunks (smaller chunk for higher recall)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

def build_chroma_if_missing(docs, persist_dir: str):
    """
    Build Chroma DB in persist_dir if missing. Returns a Chroma instance.
    We use Chroma.from_documents when building; otherwise load existing with Chroma(...).
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print(f"📦 Building Chroma DB at {persist_dir} ...")
        vectordb = Chroma.from_documents(docs, embedding_function, persist_directory=persist_dir)
        print("✅ Built Chroma DB.")
        return vectordb
    else:
        print(f"📁 Loading existing Chroma DB at {persist_dir} ...")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)

def load_chroma(persist_dir: str):
    """Return a fresh Chroma instance pointing at persist_dir."""
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)

def answer_with_rag(vectordb, query, k=5):
    """RAG pipeline using Groq LLM."""
    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
    except Exception as e:
        # Bubble up exception to caller; caller can decide to rebuild if needed
        raise

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Answer using ONLY the context below. If not found, reply 'Not found'.

Context:
{context}

Question:
{query}
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        )
        message = getattr(completion.choices[0].message, "content", None)
        return message.strip() if message else "No answer generated."
    except Exception as e:
        print(f"❌ RAG/Model error: {e}")
        return "An error occurred during generation."
