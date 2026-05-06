"""Runtime configuration.

LLM provider settings (provider, model, api_key, base_url) live in the
database (`app_settings` table) so they can be changed at runtime via the
Settings page. Only infrastructural knobs live here.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="QAI_",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    log_level: str = "INFO"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "qai.db"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def screenshots_dir(self) -> Path:
        return self.data_dir / "screenshots"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


settings = Settings()
