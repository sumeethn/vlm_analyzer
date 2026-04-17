from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    ollama_base_url: str = "http://127.0.0.1:11434"
    temp_dir: str = "/tmp/vlm_jobs"
    video_mount: str = "/data/videos"
    frame_batch_root: str = "/var/lib/nova/frame_batches"
    job_key_prefix: str = "job:"
    job_ttl_seconds: int = 86400
    caption_key_prefix: str = "caption:"
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

    frame_batch_ready_stream: str = "nova:frame_batches"
    caption_ready_stream: str = "nova:captions"
    frame_batch_stream_group: str = "vlm-captioner"
    frame_batch_stream_consumer: str = "consumer-1"
    frame_batch_stream_maxlen: int = 10_000
    max_inflight_batches_per_stream: int = 2
    max_inflight_batches_per_job: int = 8
    caption_retry_limit: int = 3
    caption_claim_idle_ms: int = 60_000
    caption_poll_block_ms: int = 5_000
    preserve_audio_artifacts: bool = False
    frame_batch_retention_seconds: int = 1800
    frame_batch_failed_retention_seconds: int = 7200
    preserve_chunk_artifacts: bool = False

    # OpenClaw event bus — set OPENCLAW_BUS_ENABLED=true to publish to Redis Streams
    openclaw_bus_enabled: bool = False
    openclaw_insights_stream: str = "openclaw:insights"
    openclaw_alerts_stream: str = "openclaw:alerts"
    openclaw_stream_maxlen: int = 10_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
