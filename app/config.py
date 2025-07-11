# app/config.py
import os

# 模型路径
MODEL_PATHS = {
    "bge": "./models/bge-reranker-base",
    "qwen4b": "./models/Qwen3-Reranker-4B",
    "qwen8b": "./models/Qwen3-Reranker-8B"
}

# Qwen3 模型的提示模板
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