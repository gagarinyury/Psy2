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

    # OpenTelemetry
    otel_exporter: str = "none"

    # RAG Vector Search Settings
    RAG_USE_VECTOR: bool = False  # Использовать векторный поиск вместо metadata
    RAG_TOP_K: int = 3  # Количество топ результатов для векторного поиска

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
