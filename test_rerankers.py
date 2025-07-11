# test_rerankers.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 测试脚本：验证本地模型的推理与QPS性能
"""
单卡测试脚本：依次验证 bge-reranker-base、Qwen3-Reranker-4B、Qwen3-Reranker-8B 
在各自单张 GPU 上的推理与 QPS 性能。

目录结构：
  rerank_models/
  ├─ models -> /media/inspur/rerank_models/models
  │    ├─ bge-reranker-base/
  │    ├─ Qwen3-Reranker-4B/
  │    └─ Qwen3-Reranker-8B/
  └─ test_rerankers.py  <- 本脚本

依赖：
    pip install torch sentence-transformers modelscope

运行：
    python test_rerankers.py
"""
import time
import torch
from sentence_transformers import CrossEncoder
from modelscope import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# GPU 分配: 每个模型固定使用一张 GPU
# ----------------------------
DEVICE_BGE     = torch.device("cuda:0")  # bge-reranker-base
DEVICE_QWEN4B  = torch.device("cuda:0")  # Qwen3-Reranker-4B
DEVICE_QWEN8B  = torch.device("cuda:1")  # Qwen3-Reranker-8B

# ----------------------------
# GPU 监控和验证函数
# ----------------------------
def check_gpu_availability():
    """检查GPU可用性和CUDA环境"""
    print("🔍 GPU环境检查:")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    print()

def get_gpu_memory(device):
    """获取指定GPU的显存使用情况"""
    if isinstance(device, torch.device):
        device_id = device.index
    else:
        device_id = device
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        return allocated, reserved
    return 0, 0

def print_gpu_usage(device, stage=""):
    """打印GPU使用情况"""
    allocated, reserved = get_gpu_memory(device)
    device_id = device.index if isinstance(device, torch.device) else device
    print(f"   📊 GPU {device_id} {stage}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

def verify_model_on_gpu(model, device):
    """验证模型是否真正在GPU上"""
    device_id = device.index if isinstance(device, torch.device) else device
    
    # 检查模型参数所在设备
    model_devices = set()
    for param in model.parameters():
        model_devices.add(param.device)
    
    print(f"   ✅ 模型参数所在设备: {model_devices}")
    
    # 检查是否在指定GPU上
    expected_device = torch.device(f"cuda:{device_id}")
    if expected_device in model_devices:
        print(f"   ✅ 模型已正确加载到 GPU {device_id}")
        return True
    else:
        print(f"   ❌ 模型未在 GPU {device_id} 上！")
        return False

# ----------------------------
# 模型路径
# ----------------------------
BGE_MODEL   = "./models/bge-reranker-base"
QWEN3_4B    = "./models/Qwen3-Reranker-4B"
QWEN3_8B    = "./models/Qwen3-Reranker-8B"

# ----------------------------
# Qwen3-Reranker 前缀与后缀固定字符串
# ----------------------------
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n"
    "</think>\n\n"
)
MAX_LEN = 8192

# ----------------------------
# bge-reranker-base 测试
# ----------------------------
def test_bge(pairs):
    print(f"=== Testing bge-reranker-base on {DEVICE_BGE} ===")
    # 记录加载前的GPU状态
    print_gpu_usage(DEVICE_BGE, "加载前")
    
    # 加载模型
    model = CrossEncoder(BGE_MODEL, device=DEVICE_BGE)
    
    # 验证模型是否在GPU上
    verify_model_on_gpu(model.model, DEVICE_BGE)
    print_gpu_usage(DEVICE_BGE, "加载后")
    
    # 推理测试
    print("🚀 开始推理...")
    start = time.perf_counter()
    scores = model.predict(pairs)
    end = time.perf_counter()
    
    print_gpu_usage(DEVICE_BGE, "推理后")
    
    for (q, d), s in zip(pairs, scores):
        print(f"[bge] “{q}” ↔ “{d}” ⇒ {s:.4f}")
    print(f"bge-reranker-base QPS: {len(pairs)/(end-start):.1f}\n")

# ----------------------------
# Qwen3-Reranker 测试
# ----------------------------
def test_qwen(name, model_path, device, pairs):
    print(f"=== Testing {name} on {device} ===")
    
    # 记录加载前的GPU状态
    print_gpu_usage(device, "加载前")
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    ).to(device).eval()
    
    # 验证模型是否在GPU上
    verify_model_on_gpu(model, device)
    print_gpu_usage(device, "加载后")
    
    # 获取 yes/no token id
    false_id = tokenizer.convert_tokens_to_ids('no')
    true_id  = tokenizer.convert_tokens_to_ids('yes')

    # 构造完整文本（prefix + instruction + query + document + suffix）
    full_texts = []
    for q, d in pairs:
        text = (
            PREFIX +
            f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n" +
            f"<Query>: {q}\n" +
            f"<Document>: {d}" +
            SUFFIX
        )
        full_texts.append(text)

    # 推理测试
    print("🚀 开始推理...")
    start = time.perf_counter()
    inputs = tokenizer(
        full_texts,
        padding='longest',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    # 移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 验证输入数据是否在正确的GPU上
    for key, tensor in inputs.items():
        print(f"   📍 输入 {key} 在设备: {tensor.device}")
    
    # 推理并提取最后一 token logits
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        print(f"   📍 输出 logits 在设备: {logits.device}")
        neg = logits[:, false_id]
        pos = logits[:, true_id]
        scores = torch.nn.functional.log_softmax(
            torch.stack([neg, pos], dim=1), dim=1
        )[:, 1].exp().tolist()
    end = time.perf_counter()

    print_gpu_usage(device, "推理后")

    for (q, d), s in zip(pairs, scores):
        print(f"[{name}] “{q}” ↔ “{d}” ⇒ {s:.4f}")
    print(f"{name} QPS: {len(pairs)/(end-start):.1f}\n")

# ----------------------------
# 主流程
# ----------------------------
if __name__ == '__main__':
    check_gpu_availability()  # 检查GPU可用性
    sample_pairs = [
        ("今天天气如何？", "今天晴天适合外出游玩"),
        ("今天天气如何？", "股市波动很大"),
        ("Python是什么语言？", "Python是一种高级编程语言"),
        ("Python是什么语言？", "北京是中国的首都"),
    ]
    test_bge(sample_pairs)
    test_qwen('Qwen3-Reranker-4B', QWEN3_4B, DEVICE_QWEN4B, sample_pairs)
    test_qwen('Qwen3-Reranker-8B', QWEN3_8B, DEVICE_QWEN8B, sample_pairs)
    print("✅ All models tested.")