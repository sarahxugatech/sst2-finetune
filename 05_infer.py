"""步骤 5：加载最佳模型并提供命令行情感预测。"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sst2_utils import get_device

MODEL_DIR = "results/best_model"
LABEL_NAMES = {0: "negative", 1: "positive"}

# 为什么：模型和分词器必须来自同一保存目录，否则词元编号可能与模型词表不一致。
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = get_device()
model.to(device)
model.eval()


def predict_sentiment(text: str) -> tuple[str, float]:
    """输入英文句子，返回 positive/negative 与置信度。"""
    # 为什么：推理也必须执行与训练相同的分词，并限制长度以避免超出模型输入上限。
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 为什么：softmax（归一化指数函数）把两个类别分数转成和为 1 的概率，便于解释置信度。
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    label_id = int(probabilities.argmax().item())
    return LABEL_NAMES[label_id], float(probabilities[label_id].item())


if __name__ == "__main__":
    print(f"已加载模型，设备：{device}。输入英文句子；输入 quit 退出。")
    while True:
        text = input("> ").strip()
        if text.lower() == "quit":
            break
        if not text:
            continue
        label, confidence = predict_sentiment(text)
        print(f"{label}（置信度：{confidence:.2%}）")
