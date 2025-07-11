# app/main.py 
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from app.ranker import ModelManager

# —— data models —— 
class RerankRequest(BaseModel):
    query: str
    document: str

class SwitchModelRequest(BaseModel):
    model_name: str  # "bge", "qwen4b", "qwen8b"

# —— 应用 & 全局 —— 
app = FastAPI()
model_manager = ModelManager()
model_manager.load_model("bge")

# 1. 异步队列与线程池
inference_queue: asyncio.Queue = asyncio.Queue()
executor = ThreadPoolExecutor(max_workers=1)  # 串行推理，顺序入 GPU

# 2. 批处理 worker
async def batch_worker():
    while True:
        batch_items = []
        try:
            first_req, first_fut = await asyncio.wait_for(inference_queue.get(), timeout=0.02)
            batch_items.append((first_req.query, first_req.document, first_fut))
        except asyncio.TimeoutError:
            continue

        # 最多合 8 条
        while len(batch_items) < 8:
            try:
                req, fut = inference_queue.get_nowait()
                batch_items.append((req.query, req.document, fut))
            except asyncio.QueueEmpty:
                break

        # 触发模型批量推理
        inputs = [(q, d) for q, d, _ in batch_items]
        futures = [fut for *_, fut in batch_items]
        # 切到线程池执行
        results = await asyncio.get_event_loop().run_in_executor(
            executor,
            model_manager.predict_batch,
            inputs
        )
        # 分发结果
        for fut, score in zip(futures, results):
            fut.set_result(float(score))

# 启动 background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())

# 3. 异步 Entry Point
@app.post("/rerank")
async def rerank(req: RerankRequest):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await inference_queue.put((req, fut))
    score = await fut  # 等待结果
    return {"score": score}

@app.post("/switch_model")
async def switch_model(req: SwitchModelRequest):
    if req.model_name not in ["bge", "qwen4b", "qwen8b"]:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    try:
        model_manager.load_model(req.model_name)
        return {"message": f"Switched to model {req.model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))