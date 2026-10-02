# DistilBERT Fine-Tuning on SST-2

Fine-tune DistilBERT on SST-2 sentiment classification (a PyTorch learning project).

## Pipeline

Data → Tokenize → Train → Evaluate → Inference

The code demonstrates an end-to-end, reproducible Hugging Face training workflow on Apple Silicon with MPS. Chinese `# 为什么：` comments explain the purpose of each important PyTorch and fine-tuning step.

## Results

**Test accuracy: 90.48% (3 epochs, lr=2e-5)**

- Held-out labeled evaluation accuracy: **90.48%** (789/872 correct)
- Confusion matrix: `[[379, 49], [34, 410]]`
- Model-selection accuracy by epoch: 94.30%, 95.10%, 95.10%

GLUE SST-2's official test labels are hidden (`-1`), so local test accuracy cannot be computed on that split. To keep evaluation honest, the official labeled validation split was held out from both training and model selection and used as the final evaluation set. A fixed 5% split of the official training data was used for model selection.

One-epoch learning-rate ablation:

| Learning rate | Validation accuracy |
|---:|---:|
| `1e-5` | 93.08% |
| `2e-5` | **94.48%** |
| `1e-4` | 94.30% |

## Training Configuration

| Setting | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Epochs | 3 |
| Train / evaluation batch size | 16 / 64 |
| Learning rate | `2e-5` |

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python 01_mps_check.py
python 02_data_tokenize.py
python 03_train.py
python 04_eval.py
python 05_infer.py
python 06_ablation.py
```

Run scripts from the repository root in numerical order. Training uses the complete SST-2 training split and can take time. Enter `quit` to leave the interactive inference demo.

For the detailed Chinese technical summary, see [REPORT.md](REPORT.md).

## Key Takeaways

- Fine-tuning updates both the pretrained backbone and the randomly initialized classifier; freezing the backbone is cheaper but limits task adaptation.
- Training loss alone is misleading: it kept falling after validation accuracy had plateaued, while validation loss increased.
- Change one variable at a time in an ablation, fix the random seed, and keep model selection separate from final evaluation.
