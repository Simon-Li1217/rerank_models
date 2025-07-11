# **模型文件说明**

> **注意：** 本仓库未直接包含下列模型文件，需从 ModelScope 平台或其他指定渠道自行下载并放置在 models/ 目录中，以保证服务正常运行。

## **需下载的模型目录**

- models/bge-reranker-base
- models/Qwen3-Reranker-4B
- models/Qwen3-Reranker-8B

## **下载与放置步骤**

1. 登录 ModelScope 平台 (ModelScope CLI)：

```bash
pip install modelscope
modelscope login
```

2. 使用 ModelScope CLI 或网页界面下载完整模型文件：

```bash
modelscope download BAAI/bge-reranker-base -d models/bge-reranker-base
modelscope download Qwen/Qwen3-Reranker-4B -d models/Qwen3-Reranker-4B
modelscope download Qwen/Qwen3-Reranker-8B -d models/Qwen3-Reranker-8B
```

确保下载完成后，models/ 目录结构如下：

```
models/
├── bge-reranker-base/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── Qwen3-Reranker-4B/
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   └── ...
└── Qwen3-Reranker-8B/
    ├── config.json
    ├── model-00001-of-00005.safetensors
    └── ...
```

3. 完成后，重新启动服务即可加载并使用这些模型文件。可以本地目录运行test_rerankers.py测试模型是否被正常部署。