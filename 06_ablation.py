"""步骤 6：一次只改变学习率的消融实验。"""

import json
from pathlib import Path

from sst2_utils import build_trainer, get_device

LEARNING_RATES = (1e-5, 2e-5, 1e-4)
ROOT = Path("results/ablation")

# 为什么：ablation（消融实验）是移除或改变单一因素、观察结果变化的对照实验。
# 为什么：三组只改变学习率，其余数据、随机种子和训练设置一致，才能把准确率差异主要归因于学习率。
# 为什么：它能检验微调对更新步幅的敏感度，但单轮单种子结果不能证明所有任务上的普遍规律。
print(f"消融实验设备：{get_device()}")
results: list[dict[str, float]] = []
for learning_rate in LEARNING_RATES:
    run_dir = ROOT / f"lr_{learning_rate:g}"
    print(f"\n开始实验：learning_rate={learning_rate:g}")
    # 为什么：长实验可能被中断；已有完整 epoch 时读取 Trainer 真实日志，避免重复浪费计算。
    state_files = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
    completed_accuracy = None
    for state_file in reversed(state_files):
        state = json.loads(state_file.read_text(encoding="utf-8"))
        eval_rows = [row for row in state["log_history"] if row.get("epoch") == 1.0 and "eval_accuracy" in row]
        if eval_rows:
            completed_accuracy = float(eval_rows[-1]["eval_accuracy"])
            break

    if completed_accuracy is None:
        trainer, _ = build_trainer(str(run_dir), learning_rate=learning_rate, epochs=1)
        trainer.train()
        metrics = trainer.evaluate()
        accuracy = float(metrics["eval_accuracy"])
    else:
        accuracy = completed_accuracy
        print(f"读取已完成的真实 checkpoint：{state_file}")
    results.append({"learning_rate": learning_rate, "eval_accuracy": accuracy})
    print(f"完成：learning_rate={learning_rate:g}, eval_accuracy={accuracy:.6f}")

# 为什么：机器可读 JSON 保存真实结果，REPORT.md 和终端表格都可据此核验而无需手抄猜测。
ROOT.mkdir(parents=True, exist_ok=True)
result_path = ROOT / "ablation_results.json"
result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n学习率消融对比表")
print("| learning_rate | eval_accuracy |")
print("|---:|---:|")
for row in results:
    print(f"| {row['learning_rate']:g} | {row['eval_accuracy']:.6f} |")
print(f"结果已保存：{result_path}")
