import tempfile
from typing import List
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from main import load_pdf_text, split_text, store_in_chroma, answer_with_rag

app = FastAPI()

PERSIST_DIR = "chroma_db"


# Schemas
class HackRxRequest(BaseModel):
    documents: str  # URL
    questions: List[str]


class HackRxResponse(BaseModel):
    answers: List[str]


def download_pdf(url):
    """Download PDF from URL to a temporary file"""
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF.")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp.write(response.content)
    temp.close()
    print(temp.name)
    return temp.name


from fastapi.responses import JSONResponse
import traceback


@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    try:
        print("🚀 Starting HackRx pipeline...")

        # 1. Download and extract PDF
        pdf_path = download_pdf(request.documents)
        text = load_pdf_text(pdf_path)
        docs = split_text(text)

        # 2. Store in vector DB
        vectordb = store_in_chroma(docs, persist_directory=PERSIST_DIR)

        # 3. Answer questions
        answers = []
        for question in request.questions:
            answer = answer_with_rag(vectordb, question)
            answers.append(answer)

        return HackRxResponse(answers=answers)

    except Exception as e:
        print("🔥 ERROR OCCURRED:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
