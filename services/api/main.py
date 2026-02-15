from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_service import RAGQueryEngine

app = FastAPI(title="Fraud RAG API")
engine = RAGQueryEngine()

class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=QueryResponse)
async def ask_retrieval_augmented_generation(query: QueryRequest):
    try:
        answer = engine.ask(query.prompt, query.top_k)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)