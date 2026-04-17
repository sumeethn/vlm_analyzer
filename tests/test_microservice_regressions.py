"""
Regression tests for the two-microservice refactor.

These tests are intentionally dependency-light: pydantic_settings, tenacity,
and redis are stubbed so the suite runs in a vanilla Python environment without
a running Redis or full dependency install.  The stubs are set up *before* any
application imports so that the module-level import machinery sees them.
"""
from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path


# ── Dependency stubs ─────────────────────────────────────────────────────────

def _install_pydantic_settings_stub() -> None:
    if "pydantic_settings" in sys.modules:
        return
    stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        # Provide sensible defaults matching Settings field defaults so that
        # Settings() can be instantiated without keyword args in tests.
        _DEFAULTS: dict = {
            "redis_url": "redis://localhost:6379/0",
            "celery_broker_url": "redis://localhost:6379/1",
            "ollama_base_url": "http://127.0.0.1:11434",
            "temp_dir": "/tmp/vlm_jobs",
            "video_mount": "/data/videos",
            "frame_batch_root": "/var/lib/nova/frame_batches",
            "job_key_prefix": "job:",
            "job_ttl_seconds": 86400,
            "caption_key_prefix": "caption:",
            "caption_ttl_seconds": 172800,
            "max_chunk_seconds": 600.0,
            "min_chunk_seconds": 0.5,
            "max_sources_per_job": 32,
            "ollama_timeout_seconds": 300.0,
            "enable_nvdec": False,
            "frames_per_chunk": 1,
            "max_frames_per_chunk": 32,
            "stream_key_prefix": "stream:",
            "streams_active_set_key": "streams:active",
            "insights_global_list_key": "insights:global",
            "insights_stream_list_prefix": "insights:stream:",
            "insights_job_list_prefix": "insights:job:",
            "insights_max_per_list": 10_000,
            "frame_batch_ready_stream": "nova:frame_batches",
            "caption_ready_stream": "nova:captions",
            "frame_batch_stream_group": "vlm-captioner",
            "frame_batch_stream_consumer": "consumer-1",
            "frame_batch_stream_maxlen": 10_000,
            "max_inflight_batches_per_stream": 2,
            "max_inflight_batches_per_job": 8,
            "caption_retry_limit": 3,
            "caption_claim_idle_ms": 60_000,
            "caption_poll_block_ms": 5_000,
            "backpressure_poll_seconds": 1.0,
            "max_backpressure_wait_seconds": 1800.0,
            "preserve_audio_artifacts": False,
            "frame_batch_retention_seconds": 1800,
            "frame_batch_failed_retention_seconds": 7200,
            "preserve_chunk_artifacts": False,
            "enable_direct_chat_completions": False,
            "openclaw_bus_enabled": False,
            "openclaw_insights_stream": "openclaw:insights",
            "openclaw_alerts_stream": "openclaw:alerts",
            "openclaw_stream_maxlen": 10_000,
        }

        def __init__(self, **kwargs: object) -> None:
            for key, default in self._DEFAULTS.items():
                setattr(self, key, kwargs.get(key, default))

    def SettingsConfigDict(**kwargs: object) -> dict:
        return kwargs

    stub.BaseSettings = BaseSettings  # type: ignore[attr-defined]
    stub.SettingsConfigDict = SettingsConfigDict  # type: ignore[attr-defined]
    sys.modules["pydantic_settings"] = stub


def _install_tenacity_stub() -> None:
    if "tenacity" in sys.modules:
        return
    tenacity = types.ModuleType("tenacity")

    def retry(*args: object, **kwargs: object):
        def decorator(fn):
            return fn
        return decorator

    tenacity.retry = retry  # type: ignore[attr-defined]
    tenacity.retry_if_exception = lambda p: p  # type: ignore[attr-defined]
    tenacity.wait_exponential = lambda **kw: kw  # type: ignore[attr-defined]
    tenacity.stop_after_attempt = lambda n: n  # type: ignore[attr-defined]
    tenacity.before_sleep_log = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["tenacity"] = tenacity


def _install_redis_stub() -> None:
    if "redis" in sys.modules:
        return
    redis_stub = types.ModuleType("redis")

    class RedisError(Exception):
        pass

    class WatchError(RedisError):
        pass

    class _Exceptions:
        ResponseError = RedisError

    class Redis:
        pass

    def from_url(*args: object, **kwargs: object) -> None:
        raise RuntimeError("redis access is not available in these unit tests")

    redis_stub.RedisError = RedisError  # type: ignore[attr-defined]
    redis_stub.WatchError = WatchError  # type: ignore[attr-defined]
    redis_stub.Redis = Redis  # type: ignore[attr-defined]
    redis_stub.from_url = from_url  # type: ignore[attr-defined]
    redis_stub.exceptions = _Exceptions()  # type: ignore[attr-defined]
    sys.modules["redis"] = redis_stub


_install_pydantic_settings_stub()
_install_tenacity_stub()
_install_redis_stub()

# ── Application imports (after stubs are in place) ───────────────────────────

from app.config import Settings
from app.video_ingest.storage import FrameBatchStorage
from app.vlm_captioner.consumer import _runtime_config_for_batch
from common.contracts.caption import CaptionReadyEvent, CaptionRecord
from common.contracts.frame_batch import FrameBatchManifest, FrameBatchReadyEvent


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_manifest(
    *,
    batch_dir: Path,
    batch_id: str,
    source_id: str = "job-1",
    status: str = "ready",
    cleanup_after_ts: float,
) -> FrameBatchManifest:
    manifest = FrameBatchManifest(
        batch_id=batch_id,
        source_scope="jobs",
        source_id=source_id,
        source_type="file",
        source_uri="/data/videos/input.mp4",
        chunk_seconds=1.0,
        chunk_start_ts=0.0,
        chunk_end_ts=1.0,
        created_at=time.time() - 20,
        frame_count=1,
        sampling_fps=1.0,
        frame_paths=[],
        manifest_path=str(batch_dir / "manifest.json"),
        frames_dir=str(batch_dir / "frames"),
        status=status,
        cleanup_after_ts=cleanup_after_ts,
    )
    manifest.write_json(batch_dir / "manifest.json")
    return manifest


# ── Tests ────────────────────────────────────────────────────────────────────

class FrameBatchStorageTests(unittest.TestCase):
    def _make_storage(self, tmp: str) -> FrameBatchStorage:
        return FrameBatchStorage(Settings(frame_batch_root=tmp))

    def test_cleanup_skips_non_terminal_batch(self) -> None:
        """Batches in 'ready' status must never be cleaned up."""
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            ready_dir = storage.build_batch_dir("jobs", "job-1", "ready-batch")
            ready_dir.mkdir(parents=True, exist_ok=True)
            _make_manifest(
                batch_dir=ready_dir,
                batch_id="ready-batch",
                status="ready",
                cleanup_after_ts=time.time() - 10,  # expired
            )

            removed = storage.cleanup_expired_batches()

            self.assertEqual(removed, 0)
            self.assertTrue((ready_dir / "manifest.json").exists())

    def test_cleanup_removes_failed_batch(self) -> None:
        """Batches in 'failed' status past their cleanup_after_ts are removed."""
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            failed_dir = storage.build_batch_dir("jobs", "job-1", "failed-batch")
            failed_dir.mkdir(parents=True, exist_ok=True)
            _make_manifest(
                batch_dir=failed_dir,
                batch_id="failed-batch",
                status="failed",
                cleanup_after_ts=time.time() - 10,
            )

            removed = storage.cleanup_expired_batches()

            self.assertEqual(removed, 1)
            self.assertFalse(failed_dir.exists())

    def test_cleanup_removes_captioned_batch(self) -> None:
        """Batches in 'captioned' status past their cleanup_after_ts are removed."""
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            captioned_dir = storage.build_batch_dir("jobs", "job-1", "captioned-batch")
            captioned_dir.mkdir(parents=True, exist_ok=True)
            _make_manifest(
                batch_dir=captioned_dir,
                batch_id="captioned-batch",
                status="captioned",
                cleanup_after_ts=time.time() - 10,
            )

            removed = storage.cleanup_expired_batches()

            self.assertEqual(removed, 1)
            self.assertFalse(captioned_dir.exists())

    def test_cleanup_retains_unexpired_captioned_batch(self) -> None:
        """A captioned batch whose retention window has not passed must be kept."""
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            captioned_dir = storage.build_batch_dir("jobs", "job-1", "captioned-batch")
            captioned_dir.mkdir(parents=True, exist_ok=True)
            _make_manifest(
                batch_dir=captioned_dir,
                batch_id="captioned-batch",
                status="captioned",
                cleanup_after_ts=time.time() + 3600,  # not yet expired
            )

            removed = storage.cleanup_expired_batches()

            self.assertEqual(removed, 0)
            self.assertTrue((captioned_dir / "manifest.json").exists())

    def test_cleanup_mixed_statuses(self) -> None:
        """Only terminal+expired batches are removed; others survive."""
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            expired_ts = time.time() - 10

            for batch_id, status in [
                ("a-ready", "ready"),
                ("b-ready-2", "ready"),
                ("c-captioned", "captioned"),
                ("d-failed", "failed"),
            ]:
                d = storage.build_batch_dir("jobs", "job-1", batch_id)
                d.mkdir(parents=True, exist_ok=True)
                _make_manifest(
                    batch_dir=d,
                    batch_id=batch_id,
                    status=status,
                    cleanup_after_ts=expired_ts,
                )

            removed = storage.cleanup_expired_batches()

            self.assertEqual(removed, 2)  # captioned + failed

    def test_sanitize_source_uri_strips_credentials(self) -> None:
        sanitized = FrameBatchStorage.sanitize_source_uri("rtsp://user:pass@example.com:554/live")
        self.assertEqual(sanitized, "rtsp://example.com:554/live")

    def test_sanitize_source_uri_no_op_without_creds(self) -> None:
        uri = "rtsp://camera.example.com:554/stream1"
        self.assertEqual(FrameBatchStorage.sanitize_source_uri(uri), uri)

    def test_write_completion_raises_on_missing_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            manifest = FrameBatchManifest(
                batch_id="b1",
                source_scope="jobs",
                source_id="j1",
                source_type="file",
                source_uri="/data/videos/input.mp4",
                chunk_seconds=1.0,
                chunk_start_ts=0.0,
                chunk_end_ts=1.0,
                created_at=0.0,
                frame_count=0,
                sampling_fps=0.0,
                frame_paths=[],
                manifest_path=None,  # explicitly None
                frames_dir=tmp,
            )
            with self.assertRaises(ValueError):
                storage.write_completion(manifest, {"choices": []})


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
        self.assertIsNone(restored.stream_id)

    def test_caption_ready_event_null_completion_path(self) -> None:
        event = CaptionReadyEvent(
            batch_id="batch-2",
            model="llava",
            manifest_path="/tmp/manifest.json",
            caption_text="hello",
            created_at=0.0,
        )
        restored = CaptionReadyEvent.from_stream_fields(event.to_stream_fields())
        self.assertIsNone(restored.completion_path)

    def test_caption_record_has_no_completion_json_method(self) -> None:
        """completion_json() was a dead method — ensure it is removed."""
        record = CaptionRecord(
            batch_id="b",
            manifest_path="/tmp/m.json",
            model="llava",
            created_at=0.0,
            updated_at=0.0,
        )
        self.assertFalse(hasattr(record, "completion_json"))


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

    def test_runtime_config_falls_back_gracefully_on_missing_processing_config(self) -> None:
        """When no processing_config is in the manifest, we fall through to job/stream lookup."""
        settings = Settings()
        manifest = FrameBatchManifest(
            batch_id="batch-2",
            source_scope="jobs",
            source_id="job-1",
            source_type="file",
            source_uri="/data/videos/input.mp4",
            chunk_seconds=1.0,
            chunk_start_ts=0.0,
            chunk_end_ts=1.0,
            created_at=0.0,
            frame_count=1,
            sampling_fps=1.0,
            frame_paths=[],
            manifest_path="/tmp/manifest.json",
            frames_dir="/tmp/frames",
            # No job_id, no stream_id → should raise RuntimeError
        )
        event = FrameBatchReadyEvent.from_manifest(manifest)

        with self.assertRaises(RuntimeError):
            _runtime_config_for_batch(settings, manifest, event)


if __name__ == "__main__":
    unittest.main()
