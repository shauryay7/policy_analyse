# api.py
import os
import hashlib
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

from main import (
    load_pdf_text,
    split_text,
    build_chroma_if_missing,
    load_chroma,
    answer_with_rag,
    url_to_hash,
    PERSIST_ROOT
)

app = FastAPI()

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# Concurrency controls
MAX_CONCURRENT_PDF_JOBS = 3
pdf_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PDF_JOBS)
# Per-document locks so multiple requests for same URL don't rebuild concurrently
_doc_locks = {}

def download_pdf(url: str, timeout: int = 30) -> str:
    """Download PDF stream to a temp file and validate mimetype."""
    try:
        r = requests.get(url, stream=True, timeout=timeout)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    content_type = r.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower():
        # Save anyway so we can debug; but raise an error to caller
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        for chunk in r.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.close()
        raise HTTPException(status_code=400, detail=f"Downloaded resource is not a PDF (Content-Type: {content_type})")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    for chunk in r.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()
    print(f"📄 Downloaded PDF to {tmp.name}")
    return tmp.name

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    url = request.documents
    doc_hash = url_to_hash(url)
    persist_dir = os.path.join(PERSIST_ROOT, doc_hash)

    # ensure lock for this document
    if doc_hash not in _doc_locks:
        _doc_locks[doc_hash] = asyncio.Lock()

    # Limit number of simultaneous heavy jobs
    async with pdf_job_semaphore:
        async with _doc_locks[doc_hash]:
            # Build DB if missing; we run blocking steps in threadpool
            try:
                if os.path.exists(persist_dir) and os.listdir(persist_dir):
                    print("📁 Using existing persisted vector DB")
                    vectordb = await run_in_threadpool(load_chroma, persist_dir)
                else:
                    print("📦 No vector DB found. Building...")
                    pdf_path = await run_in_threadpool(download_pdf, url)
                    text = await run_in_threadpool(load_pdf_text, pdf_path)
                    docs = await run_in_threadpool(split_text, text)
                    # Build persistent Chroma DB
                    vectordb = await run_in_threadpool(build_chroma_if_missing, docs, persist_dir)
                    # remove downloaded file
                    try:
                        os.remove(pdf_path)
                    except Exception:
                        pass

            except HTTPException as he:
                # bubble up HTTP errors (invalid PDF etc)
                raise he
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": f"Failed to prepare vector DB: {e}"})

    # At this point vectordb is ready (fresh instance). Answer questions concurrently.
    # We will re-load a fresh Chroma instance per worker to reduce shared state risks.
    try:
        # create fresh vectordb per worker invocation (safe)
        answers = []
        async def _answer(q):
            local_vectordb = await run_in_threadpool(load_chroma, persist_dir)
            try:
                return await run_in_threadpool(answer_with_rag, local_vectordb, q)
            except Exception as e:
                # If Chroma returns InternalError, try to rebuild once
                err_text = str(e)
                if "Failed to get segments" in err_text or "InternalError" in err_text:
                    # try rebuild once
                    try:
                        print("⚠️ Chroma InternalError detected — rebuilding DB once.")
                        # Rebuild from scratch (expensive)
                        pdf_path = await run_in_threadpool(download_pdf, url)
                        text = await run_in_threadpool(load_pdf_text, pdf_path)
                        docs = await run_in_threadpool(split_text, text)
                        await run_in_threadpool(build_chroma_if_missing, docs, persist_dir)  # will overwrite
                        os.remove(pdf_path)
                        local_vectordb = await run_in_threadpool(load_chroma, persist_dir)
                        return await run_in_threadpool(answer_with_rag, local_vectordb, q)
                    except Exception as rebuild_e:
                        print("🔥 Rebuild failed:", rebuild_e)
                        return "Error: failed to rebuild vector DB."
                else:
                    print("Error answering question:", e)
                    return "Error answering question."

        answers = await asyncio.gather(*[_answer(q) for q in request.questions])
        return HackRxResponse(answers=answers)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
