"""步骤 1：检查 Python、PyTorch 与 Apple MPS 环境。"""

import platform

import torch

# 为什么：先确认运行时版本，后续出现兼容问题时才有可复现的诊断信息。
print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")

# 为什么：is_built 表示 PyTorch 含 MPS 支持，is_available 表示当前机器此刻真的能使用 Apple GPU。
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# 为什么：训练脚本优先选择 MPS；若当前进程访问不到 MPS，则明确回退 CPU 而不是假装用了 GPU。
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Default device: {device}")
