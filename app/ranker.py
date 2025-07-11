# app/ranker.py
import torch
from sentence_transformers import CrossEncoder
from modelscope import AutoTokenizer, AutoModelForCausalLM
from app.config import MODEL_PATHS, PREFIX, SUFFIX, MAX_LEN

class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name: str):
        if model_name == self.current_model_name:
            return
        model_path = MODEL_PATHS[model_name]
        if model_name == "bge":
            self.model = CrossEncoder(model_path, device=self.device)
            #打印模型所使用的GPU
            print(f"Using GPU: {self.device}")
            self.tokenizer = None
        else: #Qwen3-Ranker加载
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device).eval()     
            self.false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.current_model_name = model_name

    def predict(self, query: str, document: str):
        if self.current_model_name == "bge":
            return self.model.predict([(query, document)])[0]

        # 构造 prompt
        prompt = (
            f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
            f"<Query>: {query}\n<Document>: {document}"
        )
        input_ids = self.tokenizer(
            prompt,
            truncation="longest_first",
            max_length=MAX_LEN - len(self.prefix_tokens) - len(self.suffix_tokens),
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = self.prefix_tokens + input_ids + self.suffix_tokens
        inputs = self.tokenizer.pad(
            {"input_ids": [full_ids]}, padding=True, return_tensors="pt", max_length=MAX_LEN
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
            score_tensor = torch.nn.functional.log_softmax(
                torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1), dim=1
            )
            return score_tensor[:, 1].exp().item()
        
    def predict_batch(self, requests: list[tuple[str, str]]) -> list[float]:
        """
        接收 List[(query, document)]，返回 List[score]
        """
        if self.current_model_name == "bge":
            # CrossEncoder.predict 已支持批量
            return self.model.predict(requests)

        # 对 Qwen3 系列统一批量处理
        queries, docs = zip(*requests)
        full_texts = []
        for q, d in zip(queries, docs):
            txt = (PREFIX +
                   "<Instruct>: Given a web search query, retrieve relevant passages...\n"
                   f"<Query>: {q}\n<Document>: {d}" +
                   SUFFIX)
            full_texts.append(txt)

        inputs = self.tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
            neg = logits[:, self.false_id]
            pos = logits[:, self.true_id]
            probs = torch.nn.functional.log_softmax(
                torch.stack([neg, pos], dim=1), dim=1
            )[:, 1].exp().tolist()
        return probs