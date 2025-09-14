from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application
    app_env: str = "dev"

    # Database
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "rag_patient"
    postgres_user: str = "rag"
    postgres_password: str = "ragpass"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_IP_PER_MIN: int = 120
    RATE_LIMIT_SESSION_PER_MIN: int = 20
    RATE_LIMIT_FAIL_OPEN: bool = False

    # OpenTelemetry
    otel_exporter: str = "none"

    # RAG Vector Search Settings
    RAG_USE_VECTOR: bool = False  # Использовать векторный поиск вместо metadata
    RAG_TOP_K: int = 3  # Количество топ результатов для векторного поиска

    # DeepSeek API Settings
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: SecretStr | None = None
    DEEPSEEK_REASONING_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_MODEL: str = "deepseek-chat"
    DEEPSEEK_TIMEOUT_S: float = 6.0

    # Feature flags for LLM nodes
    USE_DEEPSEEK_REASON: bool = False  # Использовать DeepSeek для reasoning вместо stub
    USE_DEEPSEEK_GEN: bool = False  # Использовать DeepSeek для generation вместо stub

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
