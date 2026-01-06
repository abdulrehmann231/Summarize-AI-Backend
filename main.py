import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from rag_service import get_rag_service

load_dotenv()

app = FastAPI(title="Research Paper RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    session_id: str

class UrlRequest(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Research Paper RAG API is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    try:
        service = get_rag_service()
        num_chunks = await service.process_pdf(file, session_id)
        return {
            "message": "File processed successfully",
            "session_id": session_id,
            "filename": file.filename,
            "chunks": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-url")
async def upload_url(request: UrlRequest):
    session_id = str(uuid.uuid4())
    try:
        service = get_rag_service()
        num_chunks = await service.process_url(request.url, session_id)
        return {
            "message": "URL processed successfully",
            "session_id": session_id,
            "source": request.url,
            "chunks": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        service = get_rag_service()
        answer = service.ask_question(request.question, request.session_id)
        print(answer)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
