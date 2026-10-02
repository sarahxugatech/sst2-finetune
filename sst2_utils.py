"""SST-2 微调项目的共享工具。"""

from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path("data/tokenized")


def get_device() -> torch.device:
    """优先使用 Apple GPU；不可用时诚实回退到 CPU。"""
    # 为什么：MPS 能调用 Apple Silicon GPU 加速训练，但脚本也应在无 MPS 的环境中可运行。
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_tokenized_data() -> DatasetDict:
    # 为什么：所有实验复用步骤 2 固定的数据划分，避免数据变化干扰超参数比较。
    if not DATA_DIR.exists():
        raise FileNotFoundError("未找到 data/tokenized，请先运行 python 02_data_tokenize.py")
    return load_from_disk(str(DATA_DIR))


def compute_metrics(eval_pred) -> dict[str, float]:
    # 为什么：分类模型输出每个类别的分数，取分数最大的类别才是最终预测。
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def build_trainer(
    output_dir: str,
    learning_rate: float,
    epochs: int,
    logging_steps: int = 100,
) -> tuple[Trainer, AutoTokenizer]:
    """构造训练器；主训练和消融实验共用，保证除指定变量外配置一致。"""
    set_seed(42)
    data = load_tokenized_data()

    # 为什么：backbone（主干网络）从预训练权重“热启动”，已经掌握通用语言知识；
    # 为什么：二分类头则是“冷启动”的随机权重，必须从 SST-2 标签中学会正负情感，这是微调的核心。
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 为什么：动态补齐只把一个批次补到该批次最长句子，比全局固定长度更省显存和计算。
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 为什么：2e-5 一类的小学习率能逐步调整预训练权重，避免一次更新就破坏已有语言知识。
    # 为什么：warmup（预热）让学习率在前 500 步逐渐升高，降低训练刚开始时梯度不稳定的风险。
    # 为什么：weight decay（权重衰减）轻微惩罚过大的权重，帮助模型减少过拟合。
    # 为什么：训练结束加载验证准确率最高的 checkpoint（检查点），而不是盲目采用最后一轮。
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,
        seed=42,
        data_seed=42,
        report_to="none",
        save_total_limit=3,
        # 为什么：关闭逐 step 进度条只减少终端刷屏，不改变训练计算或指标。
        disable_tqdm=True,
    )

    # 为什么：Trainer 统一处理反向传播、优化器、验证和保存，学习者可聚焦微调配置与指标。
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer, tokenizer
