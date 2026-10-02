"""步骤 4：在保留的有标签测试集上评估最佳模型。"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from sst2_utils import get_device, load_tokenized_data

MODEL_DIR = "results/best_model"

# 为什么：加载步骤 3 保存的最佳模型，保证评估对象不是随机初始化模型或任意 checkpoint。
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = get_device()
model.to(device)
model.eval()

# 为什么：这个 test 是步骤 2 保留、训练和选模都没见过的官方有标签 validation，能衡量泛化能力。
data = load_tokenized_data()["test"]
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
loader = DataLoader(data.remove_columns(["sentence", "idx"]), batch_size=64, collate_fn=collator)

predictions: list[int] = []
labels: list[int] = []
# 为什么：评估不需要梯度，inference_mode 能减少内存并避免意外修改参数。
with torch.inference_mode():
    for batch in loader:
        batch_labels = batch.pop("labels")
        logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch_labels.tolist())

accuracy = accuracy_score(labels, predictions)
matrix = confusion_matrix(labels, predictions)
print(f"评估设备：{device}")
print("说明：GLUE 官方 test 无公开标签；此处 test 是预先保留的官方 validation（训练与选模均未使用）。")
print(f"Test accuracy: {accuracy:.6f}")
print("混淆矩阵（行=真实标签，列=预测标签；0=negative，1=positive）：")
print(matrix)

# 为什么：保存机器可读指标，报告可以直接引用真实评估数字而不靠手抄。
metrics_path = Path("results/eval_metrics.json")
metrics_path.write_text(
    json.dumps(
        {
            "evaluation_split": "GLUE SST-2 validation (held out from training and model selection)",
            "accuracy": float(accuracy),
            "confusion_matrix": matrix.tolist(),
            "num_examples": len(labels),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(f"评估指标已保存：{metrics_path}")

# 为什么：只看总准确率会掩盖错误模式；阅读误判原文能帮助发现讽刺、否定或模糊表达等难点。
wrong_indices = np.flatnonzero(np.asarray(predictions) != np.asarray(labels))[:10]
print("前 10 条分错样本：")
for number, index in enumerate(wrong_indices, start=1):
    print(
        f"{number}. 原文：{data[int(index)]['sentence']}\n"
        f"   预测：{predictions[int(index)]}，真实：{labels[int(index)]}"
    )

if accuracy <= 0.90:
    print("验收提醒：accuracy 未超过 90%；可能与训练轮数、学习率、随机波动或数据划分有关，未做额外调参。")
