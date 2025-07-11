# Reranker

本仓库提供一个高性能Reranker服务，使用 FastAPI 构建，支持在 **bge-reranker-base**、**Qwen3-Reranker-4B** 与 **Qwen3-Reranker-8B** 三款模型间运行时热切换。通过 **异步队列 + 批处理 Worker**架构，实现低延迟、高吞吐的批量推理。

## 功能介绍：

- **热切换模型**：无需重启服务，使用 REST 接口即可在多款重排序模型间切换。  
- **异步批量推理**：聚合请求为固定批次，利用 GPU 并行吞吐。  
- **线程隔离计算**：使用线程池将阻塞型 GPU 计算移出事件循环，保持主线程非阻塞。  
- **容器化部署**：内置` Dockerfile `和 `docker-compose.yml`，一键在 GPU 环境中启动服务。  
- **性能监控**：可输出批次大小、队列时延、GPU 显存占用和单请求延迟等日志。

## 仓库结构

```python
rerank_models/
├── app/
│   ├── config.py        # 模型路径、提示模板、最大长度配置
│   ├── main.py          # FastAPI 应用脚本（含异步队列和批处理 Worker）
│   └── ranker.py        # ModelManager：加载、切换、批量推理逻辑
├── models/              # （挂载卷）本地模型文件目录
├── test_rerankers.py    # 单 GPU 测试脚本，验证模型推理和 QPS
├── test_load.py         # 本地（如MacBook）压力测试脚本
├── Dockerfile           # 构建镜像（含 CUDA、Python、依赖、应用等）
├── docker-compose.yml   # 使用 NVIDIA runtime 启动服务
├── requirements.txt     # Python 依赖列表
└── README.md                   
```

## 环境准备

- **操作系统**：Ubuntu 20.04 或兼容版本  
- **Docker** & **Docker Compose**  
- **NVIDIA Container Toolkit**
- **模型文件**：需下载并放置于 `models/` 目录下，可参考下方说明

## 部署过程

1. **克隆仓库**：

   ```bash
   git clone https://github.com/Simon-Li1217/rerank_models.git
   cd rerank_models
   ```

2. **准备模型**：将以下文件夹拷贝到项目根目录的 `models/` 中：

   - `bge-reranker-base`  
   - `Qwen3-Reranker-4B`  
   - `Qwen3-Reranker-8B`  

3. **可选：调整 `docker-compose.yml`**  

   - 修改 `NVIDIA_VISIBLE_DEVICES` 控制使用的 GPU ；
   - 更改端口映射（`ports`）以匹配需求。  

4. **构建并启动容器**：

   ```bash
   sudo docker compose build --no-cache
   ```

   ```bash
   sudo docker-compose up -d --build
   ```

5. **验证启动状态**：

   ```bash
   sudo docker ps
   ```

   确认 `reranker-container` 正在运行并映射端口（例如 `0.0.0.0:8000->8000/tcp`）。

6. **本地（如 MacBook）访问**：

```bash
ssh -N -L 18000:localhost:8000 ljq@211.81.55.174
```

然后即可在**本地另一个终端窗口**访问接口。

## 使用示例

### 1. 切换模型

```bash
curl -X POST http://localhost:8000/switch_model \
     -H "Content-Type: application/json" \
     -d '{"model_name":"bge"}'
```

**响应**：

```json
{"message":"Switched to model bge"}
```

### 2. 重排序接口

```bash
curl -X POST http://localhost:8000/rerank \
     -H "Content-Type: application/json" \
     -d '{
       "query": "今天天气如何？",
       "document": "今天晴天适合外出游玩"
     }'
```

**响应**：

```json
{"score":0.85}
```

## 架构示意图

```text
+--------+    HTTP    +----------------+    Queue    +--------------+
| Client | ----------> | FastAPI +     |   asyncio   | Batch Worker |
|        |             | Uvicorn (1)   |   Queue     | (asyncio)    |
+--------+             +----------------+             +------+-------+
                                                               |
                                                               v
                                                       +---------------+
                                                       | ThreadPool     |
                                                       | Executor       |
                                                       +-------+-------+
                                                               |
                                                               v
                                                      +----------------+
                                                      | ModelManager   |
                                                      | predict_batch  |
                                                      +-------+--------+
                                                               |
                                                               v
                                                         +-------------+
                                                         | GPU Inference|
                                                         +-------------+
                                                               |
                                                               v
                                                         +-------------+
                                                         | Return Score|
                                                         +-------------+
```

## 性能测试

1. **单 GPU 测试**：

   ```bash
   python test_rerankers.py
   ```

   测试各模型在单 GPU 上的推理延迟、显存占用和 QPS。

2. **本机压力测试**：在本机（如MacBook）上将以下脚本另存为 `test_load.py`：

   ```python
   import time
   import requests
   from tqdm import tqdm
   from concurrent.futures import ThreadPoolExecutor, as_completed
   
   # 请求配置
   URL = "http://localhost:8000/rerank"
   HEADERS = {"Content-Type": "application/json"}
   PAYLOAD = {
       "query": "今天天气如何？",
       "document": "今天晴天适合外出游玩"
   }
   
   # 测试参数
   NUM_REQUESTS = 5000  # 请求总数
   CONCURRENT_THREADS = 200  # 并发线程数
   
   # 响应时间记录
   response_times = []
   
   def send_request():
       start = time.time()
       try:
           response = requests.post(URL, headers=HEADERS, json=PAYLOAD)
           latency = time.time() - start
           response_times.append(latency)
           return response.status_code, latency
       except Exception:
           return "ERROR", 0
   
   def run_load_test():
       print(f"开始负载测试：共 {NUM_REQUESTS} 个请求，{CONCURRENT_THREADS} 个并发线程")
       start_time = time.time()
   
       with ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
           futures = [executor.submit(send_request) for _ in range(NUM_REQUESTS)]
           for future in tqdm(as_completed(futures), total=NUM_REQUESTS, desc="处理中"):
               status, latency = future.result()
               if status != 200:
                   print(f"请求失败，状态码: {status}")
   
       duration = time.time() - start_time
       success_count = len(response_times)
   
       if success_count:
           print("\n测试结果：")
           print(f"成功请求数: {success_count}/{NUM_REQUESTS}")
           print(f"平均响应时间: {sum(response_times)/success_count:.3f} 秒")
           print(f"最大响应时间: {max(response_times):.3f} 秒")
           print(f"最小响应时间: {min(response_times):.3f} 秒")
           print(f"QPS（每秒处理请求数）: {success_count / duration:.2f}")
       else:
           print("所有请求均失败")
   
   if __name__ == "__main__":
       run_load_test()
   ```

   **运行**：

   1. 以目标服务器211.81.55.174为例，在本机（如MacBook）上新建会话，运行：

   ```bash
   ssh -N -L 18000:localhost:8000 ljq@211.81.55.174
   ```

   2. 在原会话上运行：

   ```bash
   python test_load.py
   ```

## 调优策略

- **批次策略**：在 `app/main.py` 中修改时间窗（`wait_for` 超时）和最大批次，平衡延迟与吞吐。
- **模型量化**：在 `app/ranker.py` 启用 `load_in_8bit=True`，减少显存占用。
- **监控集成**：可以接入 Prometheus / Grafana，导出 QPS、延迟、显存使用指标。
