# app/main.py
# 单线程的 FastAPI 应用，未实现多线程或异步处理
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.ranker import ModelManager

app = FastAPI()
model_manager = ModelManager()
model_manager.load_model("bge")  # 默认加载 bge

class RerankRequest(BaseModel):
    query: str
    document: str

class SwitchModelRequest(BaseModel):
    model_name: str  # "bge", "qwen4b", "qwen8b"

@app.post("/rerank")
def rerank(request: RerankRequest):
    try:
        score = model_manager.predict(request.query, request.document)
        return {"score": float(score)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch_model")
def switch_model(request: SwitchModelRequest):
    if request.model_name not in ["bge", "qwen4b", "qwen8b"]:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    try:
        model_manager.load_model(request.model_name)
        return {"message": f"Switched to model {request.model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))