"""步骤 2：下载 SST-2、建立可复现划分并批量分词。"""

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from sst2_utils import DATA_DIR, MODEL_NAME

# 为什么：GLUE/SST-2 提供统一、可复现的数据格式，避免手工下载和解析造成差异。
raw = load_dataset("glue", "sst2")
print("官方数据集样本数：")
for split in ("train", "validation", "test"):
    print(f"  {split}: {len(raw[split])}")
print(f"一条原始样本：{raw['train'][0]}")

# 为什么：GLUE 官方 test 标签不公开（值为 -1），不能计算真实准确率；因此只把它用于展示规模。
# 为什么：从原训练集固定划出 5% 做选模验证集，官方有标签的 validation 留到步骤 4 作为独立测试集。
train_validation = raw["train"].train_test_split(test_size=0.05, seed=42)
prepared = DatasetDict(
    {
        "train": train_validation["train"],
        "validation": train_validation["test"],
        "test": raw["validation"],
    }
)
print("实际管线样本数（test 为官方有标签的 validation）：")
for split in prepared:
    print(f"  {split}: {len(prepared[split])}")

# 为什么：模型不吃原始文本；tokenizer（分词器）把句子转成模型词表中的整数编号。
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    # 为什么：input_ids 是词元在词表中的编号，是模型真正接收的文本表示。
    # 为什么：attention_mask（注意力掩码）用 1 标记真实词元、0 标记补齐位置，让模型忽略 padding（补齐）。
    return tokenizer(batch["sentence"], truncation=True, max_length=128)


# 为什么：dataset.map 配合 batched=True 一批批分词，比逐条 Python 循环更高效。
tokenized = prepared.map(tokenize_batch, batched=True, desc="批量分词")
tokenized.save_to_disk(str(DATA_DIR))

# 为什么：这里只为教学展示补齐成张量；训练时会按每个批次动态补齐，节省计算。
example_batch = tokenizer(
    prepared["train"][:2]["sentence"],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)
print(f"input_ids shape: {list(example_batch['input_ids'].shape)}")
print(f"attention_mask shape: {list(example_batch['attention_mask'].shape)}")
print(f"前 20 个 token id: {example_batch['input_ids'][0, :20].tolist()}")
print("attention_mask 作用：1 表示真实词元，0 表示 padding；模型会忽略 padding 位置。")
print(f"已保存分词数据：{DATA_DIR}")
