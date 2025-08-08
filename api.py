import tempfile
import asyncio
import traceback
from typing import List

import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from main import load_pdf_text, split_text, store_in_chroma, answer_with_rag

app = FastAPI()
PERSIST_DIR = "chroma_db"

# === Schemas ===
class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]


# === Utilities ===
def download_pdf(url):
    """Download PDF from URL to a temporary file using streaming."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF.")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    for chunk in response.iter_content(chunk_size=8192):
        temp.write(chunk)
    temp.close()
    print(f"📄 PDF saved at: {temp.name}")
    return temp.name


# === Endpoint ===
@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    try:
        print("🚀 Starting HackRx pipeline...")

        # Step 1: Download & extract text
        pdf_path = await run_in_threadpool(download_pdf, request.documents)
        text = await run_in_threadpool(load_pdf_text, pdf_path)
        docs = await run_in_threadpool(split_text, text)

        # Step 2: Store chunks in ChromaDB
        vectordb = await run_in_threadpool(store_in_chroma, docs, PERSIST_DIR)

        # Step 3: Run questions in parallel
        answers = await asyncio.gather(*[
            run_in_threadpool(answer_with_rag, vectordb, question)
            for question in request.questions
        ])

        return HackRxResponse(answers=answers)

    except Exception as e:
        print("🔥 ERROR OCCURRED:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
