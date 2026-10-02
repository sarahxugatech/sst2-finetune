"""步骤 3：用 Hugging Face Trainer 完成三轮微调。"""

import json
from pathlib import Path

from transformers import AutoModelForSequenceClassification

from sst2_utils import build_trainer, get_device

OUTPUT_DIR = Path("results")

# 为什么：训练前打印设备，能确认本次日志究竟来自 MPS 还是 CPU，而不是事后猜测。
print(f"训练设备：{get_device()}")
trainer, tokenizer = build_trainer(str(OUTPUT_DIR), learning_rate=2e-5, epochs=3)

# 为什么：train 会执行前向传播、计算损失、反向传播和参数更新，完成 backbone 与分类头的联合微调。
train_result = trainer.train()

# 为什么：多轮 accuracy 可能并列；约定选最后出现的最佳轮，避免人工挑选测试结果。
eval_rows = [row for row in trainer.state.log_history if "eval_accuracy" in row]
best_accuracy = max(row["eval_accuracy"] for row in eval_rows)
last_best_step = int([row for row in eval_rows if row["eval_accuracy"] == best_accuracy][-1]["step"])
last_best_checkpoint = OUTPUT_DIR / f"checkpoint-{last_best_step}"
trainer.model = AutoModelForSequenceClassification.from_pretrained(last_best_checkpoint)

# 为什么：显式保存按上述固定规则选出的模型，推理时无需猜哪个 checkpoint 最好。
best_model_dir = OUTPUT_DIR / "best_model"
trainer.save_model(str(best_model_dir))
tokenizer.save_pretrained(str(best_model_dir))

# 为什么：把原始日志完整保存为文本，既便于学习观察，也能追溯每个真实 loss 和 accuracy。
log_history = trainer.state.log_history
log_path = OUTPUT_DIR / "training_log.txt"
with log_path.open("w", encoding="utf-8") as file:
    for record in log_history:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\n每轮训练/验证摘要（loss 是该轮内日志点的平均值）：")
for epoch in range(1, 4):
    losses = [
        row["loss"]
        for row in log_history
        if "loss" in row and epoch - 1 < float(row.get("epoch", 0)) <= epoch
    ]
    eval_rows = [row for row in log_history if "eval_accuracy" in row and round(float(row["epoch"])) == epoch]
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    accuracy = eval_rows[-1]["eval_accuracy"] if eval_rows else float("nan")
    print(f"epoch {epoch}: train loss={mean_loss:.6f}, eval accuracy={accuracy:.6f}")

print(f"训练总体指标：{train_result.metrics}")
print(f"最佳 accuracy：{best_accuracy:.6f}")
print(f"并列时选最后一轮：{last_best_checkpoint}")
print(f"最终最佳模型：{best_model_dir}")
print(f"训练日志：{log_path}")
