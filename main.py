import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

client = Groq()
persist_directory = "chroma_db"
pdf_path = "/Users/vinayak/IdeaProjects/python/AI/GenAI/HackRx/temp/policy.pdf"
rebuild = False  # Set to False to load existing DB
body = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
    ]
}


def load_pdf_text(pdf_path):
    """Extracts text from a PDF file using pdfplumber."""
    print(f"📄 Loading PDF: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if not page_text:
                print(f"⚠️  Warning: Page {i + 1} has no extractable text.")
                continue
            text += page_text + "\n"
            print(f"✅ Processed page {i + 1}/{len(pdf.pages)}")
    return text


def split_text(text, chunk_size=1000, chunk_overlap=150):
    """Splits text into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]
    print(f"🧩 Split text into {len(docs)} chunks.")
    return docs


def store_in_chroma(docs, persist_directory="chroma_db"):
    """Embeds and stores chunks in ChromaDB."""
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    print("📦 Creating vector store and embedding documents...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"✅ Stored {len(docs)} chunks in ChromaDB at '{persist_directory}'")
    return vectordb


def load_chroma(persist_directory="chroma_db"):
    """Loads existing ChromaDB if available."""
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )


def query_chroma(vectordb, query, k=3):
    """Queries ChromaDB for top-k similar chunks."""
    print(f"\n🔍 Query: {query}")
    results = vectordb.similarity_search(query, k=k)
    print("\n📚 Top Matches:")
    for i, doc in enumerate(results):
        print(f"\n--- Match {i + 1} ---\n{doc.page_content.strip()[:1000]}")


def answer_with_rag(vectordb, query, k=3):
    """Performs RAG manually using Groq SDK (non-streaming)."""
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(query)

    # Combine top-k documents into a single context string
    context = "\n\n".join([doc.page_content for doc in docs[:k]])

    prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.
    Dont use points. NO PREAMBLE.

    Context:
    {context}

    Question:
    {query}
    """

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract and return the content safely
        message = getattr(completion.choices[0].message, "content", None)
        if message is None:
            print("⚠️ No content returned from Groq:", completion)
            return "No answer could be generated."

        return message.strip()

    except Exception as e:
        print(f"❌ Error during RAG response: {e}")
        return "An error occurred while generating the answer."


def main():
    if rebuild:
        text = load_pdf_text(pdf_path)
        docs = split_text(text)
        vectordb = store_in_chroma(docs, persist_directory)
    else:
        print("📁 Loading existing ChromaDB...")
        vectordb = load_chroma(persist_directory)

    for i in body['questions']:
        query = i
        # query_chroma(vectordb, query)
        answer_with_rag(vectordb, query)


if __name__ == "__main__":
    main()
