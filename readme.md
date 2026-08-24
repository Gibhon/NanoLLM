# NanoLLM 🧠

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

A full-stack, dual-engine decoder-only Transformer framework built from the ground up for modularity, low-level architectural transparency, and production API deployment. **NanoLLM** features end-to-end data ingestion, custom tokenization, training utilities, model comparisons, and FastAPI serving containerized with Docker and Docker Compose.

---

## 🌟 Key Highlights

* **Dual Architecture Engines:**
  * **Scratch Transformer (`src/models/scratch_model.py`):** Explicit implementation of scaled dot-product multi-head attention, fused QKV linear transformations (`nn.Linear(embed_dim, embed_dim * 3)`), dynamic lower-triangular causal masking (`torch.tril`), Pre-LayerNorm block architectures, and custom feed-forward GELU networks.
  * **PyTorch Transformer (`src/models/pytorch_model.py`):** High-level implementation utilizing `nn.TransformerEncoder` APIs optimized for fast execution and hardware accelerator bindings.
* **Complete Data Engineering Pipeline (`src/data_manager/`):**
  * Automated data downloading (`download_data.py`) and text corpus preprocessing (`data_cleaner.py`).
  * Dedicated tokenizer wrapper (`tokenizer.py`) and custom PyTorch `Dataset` loaders (`dataset.py`).
* **Modular Training & Generation Pipeline (`script/`):**
  * Configurable training loops (`engine.py`) powered by centralized runtime options (`config.py`).
  * Autoregressive decoding with temperature scaling, top-$k$ candidate filtering, and context window management.
* **Production API & Orchestration:**
  * Asynchronous REST API serving (`App/app.py`) built with **FastAPI** and **Uvicorn**.
  * Multi-container setup utilizing `Dockerfile` and `compose.yaml`.
* **Empirical Benchmarking (`data/`):**
  * Visual loss and performance comparison plots (`model_comparison1.png` – `model_comparison7.png`) benchmarking both model variants during training.

---

## 📁 Repository Structure

```text
NanoLLM/
├── App/
│   └── app.py                  # FastAPI server & REST API endpoints
├── checkpoints/
│   ├── pytorch_model.pth       # Trained high-level PyTorch model weights
│   └── scratch_model.pth       # Trained scratch transformer model weights
├── data/
│   ├── data.txt                # Raw training text corpus
│   └── model_comparison[1-7].png # Empirical evaluation & loss curve plots
├── script/
│   ├── engine.py               # Model training loop & evaluation engine
│   └── run.py                  # CLI pipeline entry point for training/inference
├── src/
│   ├── data_manager/
│   │   ├── data_cleaner.py     # Text cleaning & normalization utilities
│   │   ├── dataset.py          # PyTorch Dataset & DataLoader interfaces
│   │   ├── download_data.py    # Corpus fetching utilities
│   │   └── tokenizer.py        # Tokenizer initialization & encoding wrappers
│   └── models/
│       ├── pytorch_model.py    # PyTorch-native Transformer architecture
│       └── scratch_model.py    # Custom ground-up Transformer engine
├── config.py                   # Global hyperparameters & training configs
├── Dockerfile                  # Production container definition
├── compose.yaml                # Docker Compose orchestration config
├── README.Docker.md            # Detailed container instructions
└── requirements.txt            # Python dependencies manifest