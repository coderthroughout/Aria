from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import Optional


class Settings(BaseSettings):
    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_openai_api_key: SecretStr = SecretStr("")
    azure_openai_endpoint: str = "https://omium-ai-2-resource.openai.azure.com"
    azure_openai_api_version: str = "2024-10-21"
    azure_reasoning_deployment: str = "aria-reasoning"
    azure_extraction_deployment: str = "aria-reasoning"

    # ── Tools ─────────────────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── SEC EDGAR ─────────────────────────────────────────────────────────────
    edgar_user_agent: str = "aria-research aria@omium.ai"

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_backend: str = "memory"
    postgres_url: Optional[str] = None

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "outputs"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def reasoning_model(self) -> str:
        return self.azure_reasoning_deployment

    @property
    def extraction_model(self) -> str:
        return self.azure_extraction_deployment


settings = Settings()
