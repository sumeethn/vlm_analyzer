from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

if "pydantic_settings" not in sys.modules:
    stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def SettingsConfigDict(**kwargs):
        return kwargs

    stub.BaseSettings = BaseSettings
    stub.SettingsConfigDict = SettingsConfigDict
    sys.modules["pydantic_settings"] = stub

if "tenacity" not in sys.modules:
    tenacity = types.ModuleType("tenacity")

    def retry(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def retry_if_exception(predicate):
        return predicate

    def wait_exponential(**kwargs):
        return kwargs

    def stop_after_attempt(attempts):
        return attempts

    def before_sleep_log(*args, **kwargs):
        return None

    tenacity.retry = retry
    tenacity.retry_if_exception = retry_if_exception
    tenacity.wait_exponential = wait_exponential
    tenacity.stop_after_attempt = stop_after_attempt
    tenacity.before_sleep_log = before_sleep_log
    sys.modules["tenacity"] = tenacity

if "redis" not in sys.modules:
    redis_stub = types.ModuleType("redis")

    class RedisError(Exception):
        pass

    class WatchError(RedisError):
        pass

    class _Exceptions:
        ResponseError = RedisError

    class Redis:
        pass

    def from_url(*args, **kwargs):
        raise RuntimeError("redis access is not available in these unit tests")

    redis_stub.RedisError = RedisError
    redis_stub.WatchError = WatchError
    redis_stub.Redis = Redis
    redis_stub.from_url = from_url
    redis_stub.exceptions = _Exceptions()
    sys.modules["redis"] = redis_stub

from app.config import Settings
from app.video_ingest.storage import FrameBatchStorage
from app.vlm_captioner.consumer import _runtime_config_for_batch
from common.contracts.caption import CaptionReadyEvent
from common.contracts.frame_batch import FrameBatchManifest, FrameBatchReadyEvent


class FrameBatchStorageTests(unittest.TestCase):
    def test_cleanup_only_removes_terminal_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(frame_batch_root=str(root))
            storage = FrameBatchStorage(settings)
            now = time.time() - 10

            ready_dir = storage.build_batch_dir("jobs", "job-1", "ready-batch")
            ready_dir.mkdir(parents=True, exist_ok=True)
            FrameBatchManifest(
                batch_id="ready-batch",
                source_scope="jobs",
                source_id="job-1",
                source_type="file",
                source_uri="/data/videos/input.mp4",
                chunk_seconds=1.0,
                chunk_start_ts=0.0,
                chunk_end_ts=1.0,
                created_at=now,
                frame_count=1,
                sampling_fps=1.0,
                frame_paths=[],
                manifest_path=str(ready_dir / "manifest.json"),
                frames_dir=str(ready_dir / "frames"),
                status="ready",
                cleanup_after_ts=now,
            ).write_json(ready_dir / "manifest.json")

            failed_dir = storage.build_batch_dir("jobs", "job-1", "failed-batch")
            failed_dir.mkdir(parents=True, exist_ok=True)
            FrameBatchManifest(
                batch_id="failed-batch",
                source_scope="jobs",
                source_id="job-1",
                source_type="file",
                source_uri="/data/videos/input.mp4",
                chunk_seconds=1.0,
                chunk_start_ts=0.0,
                chunk_end_ts=1.0,
                created_at=now,
                frame_count=1,
                sampling_fps=1.0,
                frame_paths=[],
                manifest_path=str(failed_dir / "manifest.json"),
                frames_dir=str(failed_dir / "frames"),
                status="failed",
                cleanup_after_ts=now,
            ).write_json(failed_dir / "manifest.json")

            removed = storage.cleanup_expired_batches()

            self.assertEqual(removed, 1)
            self.assertTrue((ready_dir / "manifest.json").exists())
            self.assertFalse(failed_dir.exists())

    def test_sanitize_source_uri_strips_credentials(self) -> None:
        sanitized = FrameBatchStorage.sanitize_source_uri("rtsp://user:pass@example.com:554/live")
        self.assertEqual(sanitized, "rtsp://example.com:554/live")


class CaptionContractTests(unittest.TestCase):
    def test_caption_ready_event_round_trip_uses_completion_path(self) -> None:
        event = CaptionReadyEvent(
            batch_id="batch-1",
            job_id="job-1",
            stream_id=None,
            source_index=0,
            chunk_index=2,
            model="llava",
            manifest_path="/tmp/manifest.json",
            caption_text="caption",
            completion_path="/tmp/completion.json",
            created_at=123.4,
        )

        restored = CaptionReadyEvent.from_stream_fields(event.to_stream_fields())

        self.assertEqual(restored.completion_path, "/tmp/completion.json")
        self.assertEqual(restored.caption_text, "caption")


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_config_prefers_manifest_snapshot(self) -> None:
        settings = Settings()
        manifest = FrameBatchManifest(
            batch_id="batch-1",
            source_scope="jobs",
            source_id="job-1",
            source_type="file",
            source_uri="/data/videos/input.mp4",
            source_index=0,
            chunk_index=0,
            chunk_seconds=1.0,
            chunk_start_ts=0.0,
            chunk_end_ts=1.0,
            created_at=0.0,
            frame_count=1,
            sampling_fps=1.0,
            frame_paths=["/tmp/frame.jpg"],
            manifest_path="/tmp/manifest.json",
            frames_dir="/tmp/frames",
            job_id="job-1",
            metadata={
                "processing_config": {
                    "model": "snapshot-model",
                    "prompt": "snapshot-prompt",
                    "ollama_options": {"temperature": 0.2},
                }
            },
        )
        event = FrameBatchReadyEvent.from_manifest(manifest)

        model, prompt, options = _runtime_config_for_batch(settings, manifest, event)

        self.assertEqual(model, "snapshot-model")
        self.assertEqual(prompt, "snapshot-prompt")
        self.assertEqual(options, {"temperature": 0.2})


if __name__ == "__main__":
    unittest.main()
