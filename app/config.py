from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    ollama_base_url: str = "http://127.0.0.1:11434"
    temp_dir: str = "/tmp/vlm_jobs"
    video_mount: str = "/data/videos"
    job_key_prefix: str = "job:"
    job_ttl_seconds: int = 86400
    max_chunk_seconds: float = 600.0
    min_chunk_seconds: float = 0.5
    max_sources_per_job: int = 32
    ollama_timeout_seconds: float = 300.0
    enable_nvdec: bool = False
    # Frames sampled per chunk window (evenly spaced in time); all are sent in one Ollama call.
    frames_per_chunk: int = 1
    max_frames_per_chunk: int = 32

    stream_key_prefix: str = "stream:"
    streams_active_set_key: str = "streams:active"
    insights_global_list_key: str = "insights:global"
    insights_stream_list_prefix: str = "insights:stream:"
    insights_job_list_prefix: str = "insights:job:"
    insights_max_per_list: int = 10_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
