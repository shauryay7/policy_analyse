import os
import uuid
import shutil
import asyncio
import tempfile
import traceback
import requests
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from main import load_pdf_text, split_text, store_in_chroma, answer_with_rag

app = FastAPI()

class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

MAX_CONCURRENT_PDF_JOBS = 3
pdf_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PDF_JOBS)

# Global lock per PDF to avoid duplicate processing
_pdf_locks = {}
_vectordb_ready = {}

def download_pdf(url: str) -> str:
    r = requests.get(url, stream=True, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF.")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    for chunk in r.iter_content(chunk_size=8192):
        tmp_file.write(chunk)
    tmp_file.close()
    print(f"📄 PDF saved: {tmp_file.name}")
    return tmp_file.name

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    cache_key = request.documents

    # Ensure a single lock per document
    if cache_key not in _pdf_locks:
        _pdf_locks[cache_key] = asyncio.Lock()

    async with pdf_job_semaphore:
        async with _pdf_locks[cache_key]:
            try:
                # If already processed, skip directly
                if cache_key in _vectordb_ready:
                    vectordb = _vectordb_ready[cache_key]
                else:
                    pdf_path = await run_in_threadpool(download_pdf, cache_key)
                    text = await run_in_threadpool(load_pdf_text, pdf_path)
                    docs = await run_in_threadpool(split_text, text)
                    temp_chroma_dir = os.path.join(tempfile.gettempdir(), f"chroma_{uuid.uuid4().hex}")
                    vectordb = await run_in_threadpool(store_in_chroma, docs, cache_key, temp_chroma_dir)
                    _vectordb_ready[cache_key] = vectordb
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    shutil.rmtree(temp_chroma_dir, ignore_errors=True)

            except Exception as e:
                print("🔥 Pipeline error:")
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

    # Answer questions concurrently (no lock needed)
    answers = await asyncio.gather(*[
        run_in_threadpool(answer_with_rag, vectordb, q) for q in request.questions
    ])
    return HackRxResponse(answers=answers)
