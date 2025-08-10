import os
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

client = Groq()
persist_directory = "chroma_db"

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

# PDF -> Chroma vector DB cache
_vectordb_cache = {}

def load_pdf_text(pdf_path):
    """Extract text from PDF using pdfplumber."""
    print(f"📄 Loading PDF: {pdf_path}")
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if not page_text:
                print(f"⚠️ Page {i + 1} has no text.")
                continue
            text_chunks.append(page_text)
            if (i + 1) % 10 == 0:
                print(f"✅ Processed {i + 1}/{len(pdf.pages)} pages")
    return "\n".join(text_chunks)

def split_text(text, chunk_size=800, chunk_overlap=100):
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

def store_in_chroma(docs, cache_key=None, persist_directory="chroma_db"):
    """Store or retrieve from cache."""
    if cache_key and cache_key in _vectordb_cache:
        print("⚡ Using cached vector DB")
        return _vectordb_cache[cache_key]

    print("📦 Creating vector store...")
    vectordb = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
    if cache_key:
        _vectordb_cache[cache_key] = vectordb
    return vectordb

def load_chroma(persist_directory="chroma_db"):
    """Load existing ChromaDB."""
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def answer_with_rag(vectordb, query, k=5):
    """Retrieve and answer query."""
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question using ONLY the factual information in the provided context.
If the answer is not found, respond with 'Not found'.

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
        print(f"❌ RAG error: {e}")
        return "An error occurred."
