import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent / "src" / "models"))
from pytorch_model import load_pytorch_model  # type:ignore
from scratch_model import load_scratch_model  # type:ignore


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt for the model.")
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=128)  # noqa: UP045
    sampling_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)  # noqa: UP045


class GenerationResponse(BaseModel):
    response: str
    tokens_generated: int
    status: str = "Success"


def load(config):
    model_config = {
        "embed_dim": config["embed_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "block_size": config["block_size"],
        "dropout": config["dropout"],
        "tokenizer": config["tokenizer"],
        "device": config["device"],
        "model_path": config["model_path"],
        "n_feedforward": config["n_feedforward"],
    }
    if config.get("type") == "pytorch":
        return load_pytorch_model(**model_config)
    return load_scratch_model(**model_config)


def create_app(model_name, config):
    if model_name not in config:
        raise ValueError(f"Model key '{model_name}' not found in configuration dict.")

    current_config = config[model_name]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"Loading model '{model_name}' into memory...")
        app.state.model = load(config=current_config)
        app.state.config = current_config
        print("Model Loaded Successfully!")

        yield

        print("Unloading the model...")
        app.state.model.clear()

    app = FastAPI(title="AI", version="1.0", lifespan=lifespan)

    @app.get("/health", status_code=status.HTTP_200_OK)
    async def check_health():
        if not getattr(app.state, "model", None):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not initialized...",
            )
        return {status: "healthy"}

    @app.post(
        "/generate", response_model=GenerationResponse, status_code=status.HTTP_200_OK
    )
    async def generate_endpoint(payload: GenerationRequest):
        try:
            model_instance = app.state.model
            prompt_tokens = current_config["tokenizer"].encode(payload.prompt)

            max_new_tokens = (
                payload.max_new_tokens
                if payload.max_new_tokens is not None
                else current_config["max_new_tokens"]
            )

            sampling_temperature = (
                payload.sampling_temperature
                if payload.sampling_temperature is not None
                else current_config["sampling_temperature"]
            )

            allowed_tokens = min(
                max_new_tokens,
                current_config["max_context_size"] - len(prompt_tokens),
            )
            if allowed_tokens <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt length leaves no context budget for generation.",
                )
            output = model_instance.generate(
                prompt=payload.prompt,
                max_new_tokens=allowed_tokens,
                sampling_temperature=sampling_temperature,
                top_k_candidate_count=current_config["top_k_candidate_count"],
                max_context_size=current_config["max_context_size"],
            )
            return GenerationResponse(response=output, tokens_generated=len(output))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    return app


BASEPATH = Path(__file__).resolve().parent.parent

sys.path.append(str(BASEPATH))
from config import Config

sys.path.append(str(BASEPATH / "src" / "data_manager"))
from data_cleaner import clean_data  # type:ignore
from tokenizer import CharTokenizer  # type:ignore

cfg = Config()

data = clean_data(BASEPATH / "data" / "data.txt")
tokenizer = CharTokenizer(data)

model = "pytorch"
config = {
    "pytorch": {
        "type": "pytorch",
        "name": "pytorch_model",
        "embed_dim": cfg.embed_dimension,
        "n_heads": cfg.n_attention_heads,
        "n_layers": cfg.n_layers,
        "block_size": cfg.block_size,
        "dropout": cfg.dropout_pytorch,
        "tokenizer": tokenizer,
        "device": cfg.device,
        "model_path": cfg.model_path / "pytorch_model.pth",
        "max_new_tokens": cfg.max_new_tokens,
        "sampling_temperature": cfg.sampling_temperature,
        "top_k_candidate_count": cfg.top_k_candidate_count,
        "max_context_size": cfg.max_context_size,
        "n_feedforward": cfg.n_feedforward,
    },
    "scratch": {
        "type": "scratch",
        "name": "scratch_model",
        "embed_dim": cfg.embed_dimension,
        "n_heads": cfg.n_attention_heads,
        "n_layers": cfg.n_layers,
        "block_size": cfg.block_size,
        "dropout": cfg.dropout_scratch,
        "tokenizer": tokenizer,
        "device": cfg.device,
        "model_path": cfg.model_path / "scratch_model.pth",
        "max_new_tokens": cfg.max_new_tokens,
        "sampling_temperature": cfg.sampling_temperature,
        "top_k_candidate_count": cfg.top_k_candidate_count,
        "max_context_size": cfg.max_context_size,
        "n_feedforward": cfg.n_feedforward,
    },
}
app = create_app(model, config)
