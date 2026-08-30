"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from typing import Any

import msgspec
import vllm.v1.engine.core as _vllm_engine_core_module
import zmq
from vllm.logger import init_logger
from vllm.multimodal.cache import MultiModalCache, ShmObjectStoreReceiverCache
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.engine import EngineCoreReadyResponse, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.tensor_ipc import TensorIpcSender
from vllm.v1.engine.utils import EngineZmqAddresses, SignalCallback
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.version import __version__ as VLLM_VERSION

from vllm_omni.distributed.omni_coordinator import create_stage_coord_client
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.stage_init_utils import (
    make_forward_context_thread_local,
    make_workspace_manager_colocation_safe,
    maybe_apply_audex_cfg_patches,
    set_death_signal,
)

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128
_LOG_INGRESS_DIAG = os.environ.get("VLLM_OMNI_LOG_HANDOFF_DIAG", "0") not in ("0", "", "false", "False")
_LOG_CORE_STEP_DIAG = os.environ.get("VLLM_OMNI_LOG_CORE_STEP_DIAG", "0") not in (
    "0",
    "",
    "false",
    "False",
)
_DIAG_STAGE_RAW = os.environ.get("VLLM_OMNI_DIAG_STAGE")
_DIAG_STAGES = (
    None
    if _DIAG_STAGE_RAW is None
    else frozenset(stage.strip() for stage in _DIAG_STAGE_RAW.split(",") if stage.strip())
)
_OUTPUT_IPC_FALLBACK_MIN_BYTES = 1 << 20


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class _SharedOutputTensorIpcSender(TensorIpcSender):
    """Send request-owned shared outputs without an encoder-side copy.

    Stage 0's model runner places the two large Thinker conditioning layers
    directly in shared storage. Other small tensors remain in regular msgpack
    frames. An unexpected large ordinary tensor retains the old copy-to-IPC
    fallback so another latent model cannot silently regress to a huge ZMQ
    frame; diagnostics expose any such fallback.
    """

    def __init__(self, queue: Any) -> None:
        super().__init__(queue)
        self._shared_bytes = 0
        self._fallback_bytes = 0
        self._send_ms = 0.0

    def new_message(self) -> None:
        super().new_message()
        self._shared_bytes = 0
        self._fallback_bytes = 0
        self._send_ms = 0.0

    def __call__(self, tensor: Any) -> dict[str, Any] | None:
        already_shared = tensor.is_shared()
        if not already_shared:
            self._fallback_bytes += int(tensor.nbytes)
            if tensor.nbytes < _OUTPUT_IPC_FALLBACK_MIN_BYTES:
                return None
        start = time.monotonic()
        result = super().__call__(tensor)
        self._send_ms += (time.monotonic() - start) * 1000.0
        if result is not None and already_shared:
            self._shared_bytes += int(tensor.nbytes)
        elif result is None and already_shared:
            self._fallback_bytes += int(tensor.nbytes)
        return result

    def message_stats(self) -> tuple[int, int, float]:
        return self._shared_bytes, self._fallback_bytes, self._send_ms


class _MaterializedShmReceiverCache:
    """Reuse SHM-deserialized processor outputs across finite requests.

    The stock SHM receiver cache deserializes every feature every time a new
    request carries its address.  Full-prompt session requests therefore pay
    that CPU cost for historical media even when the corresponding token KV is
    prefix-cached.  Keep a bounded worker-local cache of the immutable
    ``MultiModalKwargsItem`` objects, keyed by the same content hash as the
    processor cache.

    We deliberately retain the stock touch for every feature.  The SHM object
    store uses it to balance the sender's reference count, so skipping touches
    for prefix features can prevent ring-buffer reclamation.  Keeping complete
    materialized objects, rather than only suffix metadata, also preserves the
    existing M-RoPE and preemption/recompute paths.
    """

    def __init__(self, delegate: ShmObjectStoreReceiverCache, capacity_gb: float) -> None:
        self._delegate = delegate
        self._materialized = MultiModalCache.get_lru_cache(
            capacity_gb,
            MultiModalKwargsItem,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def get_and_update_features(
        self,
        mm_features: list[MultiModalFeatureSpec],
    ) -> list[MultiModalFeatureSpec]:
        # Touch the whole request before reading any item. Besides mirroring
        # eviction order, this protects all referenced SHM objects from being
        # reclaimed while the batch is materialized.
        for feature in mm_features:
            cache_key = feature.mm_hash or feature.identifier
            self.touch_receiver_cache_item(cache_key, feature.data)

        for feature in mm_features:
            cache_key = feature.mm_hash or feature.identifier
            feature.data = self.get_and_update_item(feature.data, cache_key)
        return mm_features

    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        cached = self._materialized.get(mm_hash)
        if cached is not None:
            return cached

        materialized = self._delegate.get_and_update_item(mm_item, mm_hash)
        self._materialized[mm_hash] = materialized
        return materialized

    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        self._delegate.touch_receiver_cache_item(mm_hash, mm_item)

    def clear_cache(self) -> None:
        self._materialized.clear()
        self._delegate.clear_cache()

    def materialized_cache_info(self, *, delta: bool = False) -> Any:
        return self._materialized.stat(delta=delta)


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def __init__(self, *args: Any, output_tensor_queue: Any | None = None, **kwargs: Any) -> None:
        # vLLM's stock EngineCore output path sends tensor backing buffers as
        # ZMQ multipart frames. P/D prefilling can return hundreds of MiB of
        # Talker-conditioning hidden states in one output; msgpack walking
        # those buffers monopolizes the output thread/GIL and delays the input
        # thread from admitting unrelated requests. Local stage processes get
        # a reverse torch-shm queue so ZMQ carries only small tensor handles.
        self._output_tensor_ipc_sender = (
            _SharedOutputTensorIpcSender(output_tensor_queue)
            if output_tensor_queue is not None
            else None
        )
        super().__init__(*args, **kwargs)
        self._install_materialized_mm_receiver_cache()
        if _LOG_CORE_STEP_DIAG:
            self._install_core_step_diagnostics()
        # Cache-only D imports are deliberately kept outside Scheduler.requests:
        # they allocate/cache KV blocks but never enter model execution.
        self._pd_cache_sync_jobs: dict[str, dict[str, Any]] = {}
        self._pd_prepared_cache: dict[str, Any] = {}
        # A paired finite D request may arrive before its cache-only import has
        # completed. Keep that ADD inside D's Core instead of making the
        # orchestrator wait for a utility result on D's busy output socket.
        self._pd_pending_decode_adds: dict[str, tuple[Any, int]] = {}
        self._pd_cache_sync_last_poll = 0.0
        self._vision_preencode_executor: ThreadPoolExecutor | None = None

    def collective_rpc(
        self,
        method: Any,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Keep an auxiliary-GPU vision RPC out of the P scheduling loop.

        vLLM utility calls normally execute synchronously on the EngineCore
        thread.  That is correct when the vision tower shares the P GPU, but
        defeats auxiliary placement: a slower Encoder call on the Code2Wav GPU
        would still stop P from admitting and scheduling LLM work.  EngineCore
        already understands ``Future`` utility results, so run only this
        stateless, single-flight sidecar RPC on a dedicated host thread.
        """
        if (
            method == "preencode_minicpmo45_vision"
            and os.environ.get("MINICPMO45_VISION_ENCODER_DEVICE", "").strip()
        ):
            executor = self._vision_preencode_executor
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="minicpmo-vision-sidecar",
                )
                self._vision_preencode_executor = executor
            collective_rpc = super().collective_rpc
            return executor.submit(
                collective_rpc,
                method,
                timeout,
                args,
                kwargs,
            )
        return super().collective_rpc(method, timeout, args, kwargs)

    def shutdown(self) -> None:
        executor = self._vision_preencode_executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._vision_preencode_executor = None
        super().shutdown()

    def _install_materialized_mm_receiver_cache(self) -> None:
        """Avoid repeatedly decoding full-history SHM media on stage workers."""
        driver_worker = getattr(self.model_executor, "driver_worker", None)
        mm_cache = getattr(driver_worker, "mm_receiver_cache", None)
        if driver_worker is None or not isinstance(mm_cache, ShmObjectStoreReceiverCache):
            return

        mm_config = self.vllm_config.model_config.get_multimodal_config()
        capacity_gb = float(mm_config.mm_processor_cache_gb)
        if capacity_gb <= 0:
            return

        driver_worker.mm_receiver_cache = _MaterializedShmReceiverCache(
            mm_cache,
            capacity_gb,
        )
        logger.info(
            "Enabled %.3g GiB worker-local materialized SHM media cache",
            capacity_gb,
        )

    def _install_core_step_diagnostics(self) -> None:
        """Time the gaps around vLLM's async batch-queue control flow."""
        stage_id = getattr(self.vllm_config.model_config, "stage_id", "?")
        if _DIAG_STAGES is not None and str(stage_id) not in _DIAG_STAGES:
            return

        original_execute_model = self.model_executor.execute_model
        original_sample_tokens = self.model_executor.sample_tokens
        last_req_ids: list[str] = []

        # UniProcExecutor hydrates shared-memory multimodal features before
        # entering the model runner. Time the two receiver-cache phases
        # separately so this CPU-side work is not mistaken for GPU execution.
        driver_worker = getattr(self.model_executor, "driver_worker", None)
        mm_cache = getattr(driver_worker, "mm_receiver_cache", None)
        if driver_worker is not None and mm_cache is not None:

            def apply_mm_cache(scheduler_output: Any) -> None:
                batch_started = time.monotonic()
                batch_cpu_started = time.thread_time()
                batch_req_ids: list[str] = []
                for req_data in scheduler_output.scheduled_new_reqs:
                    batch_req_ids.append(req_data.req_id)
                    features = req_data.mm_features
                    started = time.monotonic()
                    cpu_started = time.thread_time()
                    address_features = 0
                    modality_counts: dict[str, int] = {}

                    touch_started = time.monotonic()
                    touch_cpu_started = time.thread_time()
                    for feature in features:
                        cache_key = feature.mm_hash or feature.identifier
                        item = feature.data
                        if item is not None and "address" in item:
                            address_features += 1
                        modality_counts[feature.modality] = (
                            modality_counts.get(feature.modality, 0) + 1
                        )
                        mm_cache.touch_receiver_cache_item(cache_key, item)
                    touch_done = time.monotonic()
                    touch_cpu_done = time.thread_time()

                    get_started = time.monotonic()
                    get_cpu_started = time.thread_time()
                    slow_features: list[str] = []
                    for feature in features:
                        cache_key = feature.mm_hash or feature.identifier
                        feature_started = time.monotonic()
                        feature.data = mm_cache.get_and_update_item(
                            feature.data,
                            cache_key,
                        )
                        feature_ms = (time.monotonic() - feature_started) * 1000.0
                        if feature_ms >= 5.0:
                            slow_features.append(
                                f"{feature.modality}:{feature.mm_position.length}:{feature_ms:.3f}"
                            )
                    get_done = time.monotonic()
                    get_cpu_done = time.thread_time()
                    cpu_done = time.thread_time()

                    logger.info(
                        "[CORE-STEP-DIAG] event=mm-cache-request stage=%s mono=%.6f "
                        "req=%s features=%d address_features=%d modalities=%s "
                        "touch_ms=%.3f touch_cpu_ms=%.3f get_ms=%.3f get_cpu_ms=%.3f "
                        "total_ms=%.3f cpu_ms=%.3f slow_features=%s",
                        stage_id,
                        get_done,
                        req_data.req_id,
                        len(features),
                        address_features,
                        ",".join(
                            f"{modality}:{count}"
                            for modality, count in sorted(modality_counts.items())
                        ) or "-",
                        (touch_done - touch_started) * 1000.0,
                        (touch_cpu_done - touch_cpu_started) * 1000.0,
                        (get_done - get_started) * 1000.0,
                        (get_cpu_done - get_cpu_started) * 1000.0,
                        (get_done - started) * 1000.0,
                        (cpu_done - cpu_started) * 1000.0,
                        ",".join(slow_features) or "-",
                    )

                batch_done = time.monotonic()
                batch_cpu_done = time.thread_time()
                if batch_req_ids:
                    logger.info(
                        "[CORE-STEP-DIAG] event=mm-cache-batch stage=%s mono=%.6f "
                        "reqs=%s total_ms=%.3f cpu_ms=%.3f",
                        stage_id,
                        batch_done,
                        ",".join(batch_req_ids),
                        (batch_done - batch_started) * 1000.0,
                        (batch_cpu_done - batch_cpu_started) * 1000.0,
                    )

            driver_worker._apply_mm_cache = apply_mm_cache

        worker = getattr(driver_worker, "worker", None)
        if worker is not None:
            original_worker_execute_model = worker.execute_model
            model_runner = getattr(worker, "model_runner", None)
            original_runner_execute_model = (
                getattr(model_runner, "execute_model", None)
                if model_runner is not None
                else None
            )

            if callable(original_runner_execute_model):

                def runner_execute_model(
                    scheduler_output: Any,
                    *args: Any,
                    **kwargs: Any,
                ) -> Any:
                    req_ids = list(scheduler_output.num_scheduled_tokens)
                    started = time.monotonic()
                    logger.info(
                        "[CORE-STEP-DIAG] event=runner-call-enter stage=%s mono=%.6f reqs=%s",
                        stage_id,
                        started,
                        ",".join(req_ids),
                    )
                    try:
                        return original_runner_execute_model(
                            scheduler_output,
                            *args,
                            **kwargs,
                        )
                    finally:
                        done = time.monotonic()
                        logger.info(
                            "[CORE-STEP-DIAG] event=runner-call-exit stage=%s mono=%.6f "
                            "reqs=%s total_ms=%.3f",
                            stage_id,
                            done,
                            ",".join(req_ids),
                            (done - started) * 1000.0,
                        )

                model_runner.execute_model = runner_execute_model

            def worker_execute_model(
                scheduler_output: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                req_ids = list(scheduler_output.num_scheduled_tokens)
                started = time.monotonic()
                logger.info(
                    "[CORE-STEP-DIAG] event=worker-enter stage=%s mono=%.6f reqs=%s",
                    stage_id,
                    started,
                    ",".join(req_ids),
                )
                try:
                    return original_worker_execute_model(
                        scheduler_output,
                        *args,
                        **kwargs,
                    )
                finally:
                    done = time.monotonic()
                    logger.info(
                        "[CORE-STEP-DIAG] event=worker-exit stage=%s mono=%.6f "
                        "reqs=%s total_ms=%.3f",
                        stage_id,
                        done,
                        ",".join(req_ids),
                        (done - started) * 1000.0,
                    )

            driver_worker.worker.execute_model = worker_execute_model

        def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal last_req_ids
            last_req_ids = list(scheduler_output.num_scheduled_tokens)
            started = time.monotonic()
            if last_req_ids:
                logger.info(
                    "[CORE-STEP-DIAG] event=execute-enter stage=%s mono=%.6f reqs=%s",
                    stage_id,
                    started,
                    ",".join(last_req_ids),
                )
            result = original_execute_model(scheduler_output, *args, **kwargs)
            done = time.monotonic()
            if last_req_ids:
                logger.info(
                    "[CORE-STEP-DIAG] event=execute-exit stage=%s mono=%.6f reqs=%s total_ms=%.3f",
                    stage_id,
                    done,
                    ",".join(last_req_ids),
                    (done - started) * 1000.0,
                )
            return result

        def sample_tokens(*args: Any, **kwargs: Any) -> Any:
            req_ids = list(last_req_ids)
            started = time.monotonic()
            logger.info(
                "[CORE-STEP-DIAG] event=sample-enter stage=%s mono=%.6f reqs=%s",
                stage_id,
                started,
                ",".join(req_ids),
            )
            result = original_sample_tokens(*args, **kwargs)
            done = time.monotonic()
            logger.info(
                "[CORE-STEP-DIAG] event=sample-exit stage=%s mono=%.6f reqs=%s total_ms=%.3f",
                stage_id,
                done,
                ",".join(req_ids),
                (done - started) * 1000.0,
            )

            original_result = getattr(result, "result", None)
            if callable(original_result):
                def timed_result(*result_args: Any, **result_kwargs: Any) -> Any:
                    wait_started = time.monotonic()
                    logger.info(
                        "[CORE-STEP-DIAG] event=future-result-enter stage=%s mono=%.6f reqs=%s",
                        stage_id,
                        wait_started,
                        ",".join(req_ids),
                    )
                    try:
                        return original_result(*result_args, **result_kwargs)
                    finally:
                        wait_done = time.monotonic()
                        logger.info(
                            "[CORE-STEP-DIAG] event=future-result-exit stage=%s mono=%.6f "
                            "reqs=%s total_ms=%.3f",
                            stage_id,
                            wait_done,
                            ",".join(req_ids),
                            (wait_done - wait_started) * 1000.0,
                        )

                result.result = timed_result
            return result

        self.model_executor.execute_model = execute_model
        self.model_executor.sample_tokens = sample_tokens

    def start_pd_cache_sync(self, request: OmniEngineCoreRequest) -> Future:
        """Import P KV into D's prefix cache without creating a D request."""
        # Utility arguments use the generic msgpack decoder. With postponed
        # annotations, upstream's automatic msgspec conversion cannot see the
        # concrete Struct type and an array-like request arrives as a list.
        if not isinstance(request, OmniEngineCoreRequest):
            request = msgspec.convert(
                request,
                type=OmniEngineCoreRequest,
                dec_hook=MsgpackDecoder().dec_hook,
            )
        request_id = request.request_id
        if request_id in self._pd_cache_sync_jobs:
            raise RuntimeError(f"P/D cache sync already exists for {request_id}")
        scheduler_request, _ = self.preprocess_add_request(request)
        connector = self.scheduler.get_kv_connector()
        if connector is None:
            raise RuntimeError("P/D cache sync requires a configured KV connector")
        connector.on_new_request(scheduler_request)
        future: Future = Future()
        self._pd_cache_sync_jobs[request_id] = {
            "request": scheduler_request,
            "future": future,
            "phase": "queued",
            "started": time.monotonic(),
        }
        return future

    @staticmethod
    def _strip_remote_prefill_params(request: Any) -> None:
        """Turn a fallback-capable D ADD into a local-prefix admission."""
        request.kv_transfer_params = None
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None)
        if isinstance(extra_args, dict) and "kv_transfer_params" in extra_args:
            sampling_params.extra_args = dict(extra_args)
            sampling_params.extra_args.pop("kv_transfer_params", None)
        get_skip = getattr(request, "get_skip_reading_prefix_cache", None)
        if callable(get_skip):
            request.skip_reading_prefix_cache = get_skip()

    def _activate_pd_decode(
        self,
        request: Any,
        request_wave: int,
        prepared: dict[str, Any] | None,
    ) -> None:
        """Admit a formal D request after resolving its local import state."""
        if prepared is not None:
            held_ms = (time.monotonic() - prepared["ready_mono"]) * 1000.0
            prepared_request = prepared["request"]
            prepared_prompt = list(prepared_request.prompt_token_ids)
            request_prompt = list(request.prompt_token_ids)
            prefix_matches = (
                len(request_prompt) >= len(prepared_prompt)
                and request_prompt[: len(prepared_prompt)] == prepared_prompt
            )
            if not prefix_matches:
                if prepared.get("owns_blocks", False):
                    self.scheduler.release_direct_pd_cache_sync(prepared_request)
                logger.warning(
                    "Prepared P/D prefix changed before activation for %s; "
                    "falling back to the request's ordinary remote-prefill path",
                    request.request_id,
                )
                super().add_request(request, request_wave)
                return
            self._strip_remote_prefill_params(request)
            # A loading import owns the exact (possibly non-block-aligned)
            # block table under this request id. A full local hit owns no
            # private table and should be matched by normal prefix caching.
            if prepared.get("owns_blocks", False):
                request.num_computed_tokens = prepared_request.num_computed_tokens
            if _LOG_INGRESS_DIAG:
                logger.info(
                    "[PD-D-CONTROL] event=prepared-cache-activate request=%s "
                    "held_ms=%.3f imported_tokens=%d owns_blocks=%s",
                    request.request_id,
                    held_ms,
                    request.num_computed_tokens,
                    bool(prepared.get("owns_blocks", False)),
                )
        super().add_request(request, request_wave)

    def _activate_pending_pd_decode(
        self,
        request_id: str,
        prepared: dict[str, Any] | None,
    ) -> bool:
        pending = self._pd_pending_decode_adds.pop(request_id, None)
        if pending is None:
            return False
        request, request_wave = pending
        self._activate_pd_decode(request, request_wave, prepared)
        return True

    def add_request(self, request: Any, request_wave: int = 0) -> None:
        """Resolve a paired early import locally without an API-side barrier."""
        request_id = request.request_id
        prepared = self._pd_prepared_cache.pop(request_id, None)
        if prepared is not None:
            self._activate_pd_decode(request, request_wave, prepared)
            return

        if request_id in self._pd_cache_sync_jobs:
            if request_id in self._pd_pending_decode_adds:
                raise RuntimeError(f"Duplicate pending P/D decode ADD for {request_id}")
            self._pd_pending_decode_adds[request_id] = (request, request_wave)
            if _LOG_INGRESS_DIAG:
                logger.info(
                    "[PD-D-CONTROL] event=decode-held-for-import request=%s",
                    request_id,
                )
            return

        # The early import was unavailable or failed before this ADD reached
        # D. Its remote-prefill parameters provide the ordinary safe fallback.
        super().add_request(request, request_wave)

    def abort_requests(self, request_ids: list[str]) -> None:
        """Also release retained early imports that never reach D decode."""
        for request_id in request_ids:
            # A transfer already submitted to NIXL cannot be synchronously
            # revoked safely.  Let it finish as disposable cache instead of
            # retaining a formal-query block table whose ADD was cancelled.
            job = self._pd_cache_sync_jobs.get(request_id)
            if job is not None:
                job["request"].pd_cache_sync_retain = False
            prepared = self._pd_prepared_cache.pop(request_id, None)
            if prepared is not None:
                if prepared.get("owns_blocks", False):
                    self.scheduler.release_direct_pd_cache_sync(prepared["request"])
            self._pd_pending_decode_adds.pop(request_id, None)
        super().abort_requests(request_ids)

    def has_work(self) -> bool:
        return super().has_work() or bool(self._pd_cache_sync_jobs)

    def _fail_pd_cache_sync_job(self, request_id: str, exc: BaseException) -> None:
        job = self._pd_cache_sync_jobs.pop(request_id, None)
        if job is None:
            return
        request = job["request"]
        if job["phase"] == "loading":
            try:
                self.scheduler.fail_direct_pd_cache_sync(request)
            except Exception:
                logger.exception(
                    "Failed to release direct P/D cache sync %s", request_id
                )
        # The formal ADD retained its ordinary remote-prefill parameters. Once
        # the failed cache-only request has released connector state, admit it
        # through vLLM's standard P/D path without another API round trip.
        try:
            self._activate_pending_pd_decode(request_id, prepared=None)
        except Exception as fallback_exc:
            exc = fallback_exc
        future = job["future"]
        if not future.done():
            future.set_exception(exc)

    def _progress_pd_cache_sync_jobs(self) -> bool:
        if not self._pd_cache_sync_jobs:
            return False
        progressed = False

        # Preserve revision order inside each lineage. Normal prefix-cache
        # matching only sees completed imports; registering a newer revision
        # first would therefore request a redundant large suffix and increase
        # D cache pressure. Independent lineages may still progress in the
        # same Core pass below.
        lineage_heads: dict[str, str] = {}
        for candidate_id, candidate in self._pd_cache_sync_jobs.items():
            candidate_request = candidate["request"]
            lineage_id = getattr(candidate_request, "kv_lineage_id", None)
            if not lineage_id:
                continue
            current_id = lineage_heads.get(lineage_id)
            if current_id is None:
                lineage_heads[lineage_id] = candidate_id
                continue
            current_request = self._pd_cache_sync_jobs[current_id]["request"]
            candidate_key = (
                int(getattr(candidate_request, "kv_lineage_revision", 0)),
                float(candidate["started"]),
                candidate_id,
            )
            current_key = (
                int(getattr(current_request, "kv_lineage_revision", 0)),
                float(self._pd_cache_sync_jobs[current_id]["started"]),
                current_id,
            )
            if candidate_key < current_key:
                lineage_heads[lineage_id] = candidate_id

        # Admission and registration are scheduler-owned and therefore run on
        # this EngineCore thread. Worker RPC only starts the NIXL background
        # writer; it performs no model forward. Keep each RPC request-sized:
        # bulk registration allocates many long-context D block tables at once
        # and can evict the prefixes the following requests need.
        for request_id, job in list(self._pd_cache_sync_jobs.items()):
            if job["phase"] != "queued":
                continue
            request = job["request"]
            lineage_id = getattr(request, "kv_lineage_id", None)
            if lineage_id and lineage_heads.get(lineage_id) != request_id:
                continue
            try:
                phase, metadata = self.scheduler.prepare_direct_pd_cache_sync(
                    request
                )
                if phase == "blocked":
                    continue
                # Mark allocated imports before the worker RPC so an RPC
                # failure releases their D block table instead of leaking it.
                job["phase"] = phase
                self.model_executor.collective_rpc(
                    "start_pd_cache_sync",
                    args=(metadata,),
                )
                progressed = True
                if phase == "full_hit":
                    ready_mono = time.monotonic()
                    elapsed_ms = (ready_mono - job["started"]) * 1000.0
                    self._pd_cache_sync_jobs.pop(request_id, None)
                    prepared_cache = None
                    if bool(getattr(job["request"], "pd_cache_sync_retain", False)):
                        prepared_cache = {
                            "request": job["request"],
                            "ready_mono": ready_mono,
                            "owns_blocks": False,
                        }
                        if not self._activate_pending_pd_decode(
                            request_id,
                            prepared_cache,
                        ):
                            self._pd_prepared_cache[request_id] = prepared_cache
                    job["future"].set_result(
                        {
                            "request_id": request_id,
                            "cache_sync_ms": elapsed_ms,
                            "full_hit": True,
                        }
                    )
                else:
                    job["phase"] = "loading"
            except Exception as exc:
                self._fail_pd_cache_sync_job(request_id, exc)

        loading = {
            request_id
            for request_id, job in self._pd_cache_sync_jobs.items()
            if job["phase"] == "loading"
        }
        now = time.monotonic()
        if not loading or now - self._pd_cache_sync_last_poll < 0.001:
            return progressed
        self._pd_cache_sync_last_poll = now

        try:
            rank_results = self.model_executor.collective_rpc("poll_pd_cache_sync")
            finished_by_rank = [set(result or ()) for result in rank_results]
            finished = (
                set.intersection(*finished_by_rank) if finished_by_rank else set()
            )
        except Exception as exc:
            for request_id in loading:
                self._fail_pd_cache_sync_job(request_id, exc)
            return True

        for request_id in finished & loading:
            job = self._pd_cache_sync_jobs.get(request_id)
            if job is None:
                continue
            try:
                self.scheduler.complete_direct_pd_cache_sync(job["request"])
                elapsed_ms = (time.monotonic() - job["started"]) * 1000.0
                self._pd_cache_sync_jobs.pop(request_id, None)
                if bool(getattr(job["request"], "pd_cache_sync_retain", False)):
                    prepared_cache = {
                        "request": job["request"],
                        "ready_mono": time.monotonic(),
                        "owns_blocks": True,
                    }
                    if not self._activate_pending_pd_decode(
                        request_id,
                        prepared_cache,
                    ):
                        self._pd_prepared_cache[request_id] = prepared_cache
                job["future"].set_result(
                    {
                        "request_id": request_id,
                        "cache_sync_ms": elapsed_ms,
                        "full_hit": False,
                    }
                )
                if _LOG_INGRESS_DIAG:
                    logger.info(
                        "[PD-D-CONTROL] event=completion-observed request=%s total_ms=%.3f",
                        request_id,
                        elapsed_ms,
                    )
                progressed = True
            except Exception as exc:
                self._fail_pd_cache_sync_job(request_id, exc)
        return progressed

    def _process_engine_step(self) -> bool:
        # Preserve ordinary inference priority. Cache-only work is progressed
        # between scheduler/model steps and when D would otherwise be idle.
        base_has_work = super().has_work()
        model_executed = super()._process_engine_step() if base_has_work else False
        cache_progressed = self._progress_pd_cache_sync_jobs()
        if not base_has_work and not cache_progressed and self._pd_cache_sync_jobs:
            time.sleep(0.001)
        return model_executed or cache_progressed

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ) -> None:
        """Send outputs with out-of-band local tensor IPC when available."""
        encoder = MsgpackEncoder(oob_tensor_consumer=self._output_tensor_ipc_sender)
        reuse_buffers: list[bytearray] = []
        pending = deque()

        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(make_zmq_socket(ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000))
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Stage-0's large latent tensors already travel through the
                # reverse torch-shm queue.  What remains in the ZMQ message is
                # a small control frame (plus, at most, small ordinary tensor
                # frames).  Do not reuse a zero-copy ``bytearray`` for this
                # path: under a sustained output rate pyzmq can report the
                # tracker complete while a queued frame still observes a
                # later ``encode_into`` mutation, leaving a valid msgpack
                # object followed by stale bytes.  A fresh, copied control
                # message is cheap here and gives the receiver stable frame
                # ownership.  Stages without reverse tensor IPC retain
                # vLLM's zero-copy/reuse path below.
                if self._output_tensor_ipc_sender is not None:
                    encode_start = time.monotonic()
                    buffers = encoder.encode(outputs)
                    encode_done = time.monotonic()
                    stage_id = getattr(
                        self.vllm_config.model_config,
                        "stage_id",
                        "?",
                    )
                    if _LOG_INGRESS_DIAG and (
                        _DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES
                    ):
                        req_ids = ",".join(
                            str(getattr(item, "request_id", "?"))
                            for item in getattr(outputs, "outputs", ())
                        )
                        shared_bytes, fallback_bytes, ipc_send_ms = (
                            self._output_tensor_ipc_sender.message_stats()
                        )
                        logger.info(
                            "[HANDOFF-DIAG] event=core-output-encoded stage=%s "
                            "wall=%.6f reqs=%s encode_ms=%.3f ipc_send_ms=%.3f "
                            "shared_mib=%.3f fallback_mib=%.3f frames=%d",
                            stage_id,
                            time.time(),
                            req_ids,
                            (encode_done - encode_start) * 1000.0,
                            ipc_send_ms,
                            shared_bytes / float(1 << 20),
                            fallback_bytes / float(1 << 20),
                            len(buffers),
                        )
                    sockets[client_index].send_multipart(buffers, copy=True)
                    continue

                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                encode_start = time.monotonic()
                buffers = encoder.encode_into(outputs, buffer)
                encode_done = time.monotonic()
                stage_id = getattr(self.vllm_config.model_config, "stage_id", "?")
                if _LOG_INGRESS_DIAG and (_DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES):
                    req_ids = ",".join(
                        str(getattr(item, "request_id", "?"))
                        for item in getattr(outputs, "outputs", ())
                    )
                    shared_bytes = fallback_bytes = 0
                    ipc_send_ms = 0.0
                    if self._output_tensor_ipc_sender is not None:
                        (
                            shared_bytes,
                            fallback_bytes,
                            ipc_send_ms,
                        ) = self._output_tensor_ipc_sender.message_stats()
                    logger.info(
                        "[HANDOFF-DIAG] event=core-output-encoded stage=%s wall=%.6f "
                        "reqs=%s encode_ms=%.3f ipc_send_ms=%.3f shared_mib=%.3f "
                        "fallback_mib=%.3f frames=%d",
                        stage_id,
                        time.time(),
                        req_ids,
                        (encode_done - encode_start) * 1000.0,
                        ipc_send_ms,
                        shared_bytes / float(1 << 20),
                        fallback_bytes / float(1 << 20),
                        len(buffers),
                    )
                tracker = sockets[client_index].send_multipart(buffers, copy=False, track=True)
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    reuse_buffers.append(buffer)

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ) -> None:
        """Receive requests and split socket receive from request decoding.

        This mirrors vLLM's input thread. The only behavioral difference is
        optional timing around ``recv_multipart`` and ``MsgpackDecoder`` so a
        pre-scheduler ingress tail is not incorrectly attributed to GPU queue
        time. The diagnostic path only walks frame descriptors; it does not
        copy tensor payloads.
        """
        add_request_decoder = MsgpackDecoder(
            OmniEngineCoreRequest,
            oob_tensor_provider=self.tensor_ipc_receiver,
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        input_address,
                        zmq.DEALER,
                        identity=identity,
                        bind=False,
                    )
                )
                for input_address in input_addresses
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                if coord_input_address is not None
                else None
            )
            if coord_socket is not None:
                coord_socket.send(b"\x01")

            poller = zmq.Poller()
            ready_response = EngineCoreReadyResponse(
                max_model_len=self.vllm_config.model_config.max_model_len,
                num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
                block_size=self.vllm_config.cache_config.block_size,
                dp_stats_address=self.frontend_stats_publish_address,
                dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
                vllm_version=VLLM_VERSION,
                world_size=self.vllm_config.parallel_config.world_size,
                data_parallel_size=self.vllm_config.parallel_config.data_parallel_size,
                kv_cache_size_tokens=self.vllm_config.cache_config.kv_cache_size_tokens,
                kv_cache_max_concurrency=self.vllm_config.cache_config.kv_cache_max_concurrency,
            )
            ready_payload = msgspec.msgpack.encode(ready_response)
            for input_socket in input_sockets:
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    recv_start = time.monotonic()
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    recv_done = time.monotonic()
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    if request_type == EngineCoreRequestType.ADD:
                        req = add_request_decoder.decode(data_frames)
                        decode_done = time.monotonic()
                        stage_id = getattr(self.vllm_config.model_config, "stage_id", "?")
                        if _LOG_INGRESS_DIAG and (_DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES):
                            wire_bytes = sum(frame.buffer.nbytes for frame in data_frames)
                            logger.info(
                                "[INGRESS-DIAG] event=core-receive-decode stage=%s wall=%.6f req=%s "
                                "recv_ms=%.3f decode_ms=%.3f frames=%d wire_mib=%.3f",
                                stage_id,
                                time.time(),
                                req.request_id,
                                (recv_done - recv_start) * 1000.0,
                                (decode_done - recv_done) * 1000.0,
                                len(data_frames),
                                wire_bytes / float(1 << 20),
                            )
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)
                        if request_type == EngineCoreRequestType.ABORT:
                            self.aborts_queue.put_nowait(request)

                    self.input_queue.put_nowait((request_type, request))

    def preprocess_add_request(self, request: OmniEngineCoreRequest) -> tuple[Any, int]:
        """Preserve omni payloads when vLLM builds its scheduler request."""
        stage_id = getattr(
            getattr(getattr(self, "vllm_config", None), "model_config", None),
            "stage_id",
            "?",
        )
        ingress_diag = _LOG_INGRESS_DIAG and (_DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES)
        ingress_start = time.monotonic() if ingress_diag else 0.0
        prepare_lineage = getattr(self.scheduler, "prepare_kv_lineage_request", None)
        if prepare_lineage is not None:
            prepare_lineage(request)
        lineage_done = time.monotonic() if ingress_diag else 0.0
        # D already receives the complete prompt KV from P. Its lightweight
        # mm_features exist only so Qwen can reconstruct M-RoPE positions.
        # Bypass upstream's D-local media-cache lookup, then restore the
        # position metadata on the scheduler request for the model runner.
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None)
        kv_transfer_params = extra_args.get("kv_transfer_params") if isinstance(extra_args, dict) else None
        receives_remote_pd_kv = isinstance(kv_transfer_params, dict) and kv_transfer_params.get(
            "do_remote_prefill"
        ) is True
        pd_mm_features = None
        if (
            (getattr(request, "pd_prefill_payload", None) is not None or receives_remote_pd_kv)
            and request.mm_features
        ):
            pd_mm_features = request.mm_features
            request.mm_features = []
        try:
            scheduler_request, current_wave = super().preprocess_add_request(request)
        finally:
            if pd_mm_features is not None:
                request.mm_features = pd_mm_features
        if pd_mm_features is not None:
            scheduler_request.mm_features = pd_mm_features
        scheduler_request.additional_information = request.additional_information
        scheduler_request.model_intermediate_buffer = getattr(request, "model_intermediate_buffer", None)
        scheduler_request.pd_prefill_payload = getattr(request, "pd_prefill_payload", None)
        scheduler_request.pd_cache_sync_retain = bool(
            getattr(request, "pd_cache_sync_retain", False)
        )
        scheduler_request.external_req_id = getattr(request, "external_req_id", request.request_id)
        if ingress_diag:
            ingress_done = time.monotonic()
            logger.info(
                "[INGRESS-DIAG] event=core-preprocess stage=%s wall=%.6f req=%s "
                "prompt=%d lineage_ms=%.3f request_build_ms=%.3f total_ms=%.3f",
                stage_id,
                time.time(),
                request.request_id,
                len(request.prompt_token_ids),
                (lineage_done - ingress_start) * 1000.0,
                (ingress_done - lineage_done) * 1000.0,
                (ingress_done - ingress_start) * 1000.0,
            )
        return scheduler_request, current_wave

    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process.

        Omni-specific kwargs:
          - ``omni_coordinator_address``: ROUTER address of the head-side
            :class:`OmniCoordinator`. When provided, this subprocess
            instantiates an :class:`OmniCoordClientForStage` after the
            HELLO/INIT/READY handshake completes and reports its status +
            queue length via heartbeats. The hook is wired so each
            heartbeat refreshes ``queue_length`` from the live scheduler.
          - ``omni_stage_id``: logical stage id this replica belongs to.
            Required when ``omni_coordinator_address`` is provided.
          - ``omni_replica_id``: cluster-unique replica id within the
            stage (assigned by :class:`OmniMasterServer`). Used for
            logging / metrics only.
        """
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()

        # Colocated sibling stage(s): further engine cores built in THIS
        # process after the primary's, so all stages share one CUDA context
        # and their kernels can overlap instead of time-slicing. Popped before
        # the primary's construction -- not an EngineCoreProc argument. A bare
        # dict (legacy single-guest form) is normalized to a one-item list.
        sibling_stage_kwargs: Any = kwargs.pop("sibling_stage_kwargs", None)
        if isinstance(sibling_stage_kwargs, dict):
            sibling_stage_kwargs = [sibling_stage_kwargs]

        # Register vllm-omni reasoning parsers (e.g. step_audio) in this
        # subprocess so they are available when the engine core resolves
        # ``--reasoning-parser``.  The main process already registered them
        # at import time, but the forked subprocess starts with a fresh
        # ReasoningParserManager.
        try:
            import vllm_omni.reasoning  # noqa: F401
        except ImportError:
            logger.warning(
                "Failed to import vllm_omni.reasoning in subprocess; "
                "custom reasoning parsers (e.g. step_audio) will not be "
                "available."
            )

        engine_core: StageEngineCoreProc | None = None
        coord_client = None
        try:
            # NOTE: previous revisions hardcoded data_parallel_size=1 here
            # (TODO referencing issue #984). The hardcoding has been removed
            # so the DP fields propagate through from the caller exactly
            # like upstream vLLM.

            stage_label = f"stage{omni_stage_id}" if omni_stage_id is not None else "noid"
            set_death_signal(signal.SIGTERM)
            set_process_title(f"StageEngineCoreProc_{stage_label}_replica{omni_replica_id}_DP{dp_rank}")
            decorate_logs()
            # Workaround for flashinfer/jit-cache version mismatch in CI.
            # The parent process handles this gracefully via ring_globals.py,
            # but the subprocess hits an unprotected import in TopKTopPSampler.
            # Setting this env var allows the same graceful fallback to work.
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            os.environ["VLLM_OMNI_REPLICA_ID"] = str(max(int(omni_replica_id), 0))

            # Patch the decoder type so process_input_sockets (started
            # during __init__) decodes OmniEngineCoreRequest (which
            # carries additional_information) instead of the base
            # EngineCoreRequest.  Must happen BEFORE __init__ because
            # the IO thread creates MsgpackDecoder(EngineCoreRequest)
            # during __init__.
            _vllm_engine_core_module.EngineCoreRequest = OmniEngineCoreRequest
            logger.debug(
                "[StageEngineCoreProc] Patched EngineCoreRequest -> OmniEngineCoreRequest: %s",
                _vllm_engine_core_module.EngineCoreRequest,
            )

            # Audex CFG scheduler patches must land before EngineCore builds
            # its Scheduler; gated on the stage's logits_processors config.
            maybe_apply_audex_cfg_patches(kwargs.get("vllm_config"))

            if sibling_stage_kwargs is not None:
                # Two engine cores will step concurrently from two threads;
                # vllm's module-global forward context must become
                # thread-local BEFORE either core runs a forward.
                make_forward_context_thread_local()
                # And the guest's init must not replace-and-lock the shared
                # MoE workspace singleton out from under the host.
                make_workspace_manager_colocation_safe()

            # With a colocated sibling, the HOST must leave the legacy default
            # stream too: the default stream synchronizes with every other
            # stream, so host kernels left on it would serialize against the
            # sibling's private stream and no overlap could ever happen.
            host_stream = None
            if sibling_stage_kwargs is not None:
                import torch as _torch_host

                host_stream = _torch_host.cuda.Stream()
                _host_stream_ctx: Any = _torch_host.cuda.stream(host_stream)
            else:
                _host_stream_ctx = contextlib.nullcontext()

            with _host_stream_ctx:
                engine_core = StageEngineCoreProc(
                    *args,
                    engine_index=dp_rank,
                    **kwargs,
                )

            # Each subprocess corresponds to exactly one omni replica with
            # its own OmniMasterServer allocation, so the heartbeat client
            # runs unconditionally — there is no dp_rank-based gating.
            if omni_coordinator_address is not None:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coordinator_address is set")
                addresses: EngineZmqAddresses = engine_core.addresses
                if not addresses.inputs or not addresses.outputs:
                    raise RuntimeError(
                        "EngineCore handshake did not populate input/output addresses; "
                        "cannot start OmniCoordClientForStage"
                    )
                scheduler = getattr(engine_core, "scheduler", None)
                if scheduler is None:
                    raise RuntimeError("EngineCore scheduler is not initialized")
                coord_client = create_stage_coord_client(
                    coord_zmq_addr=omni_coordinator_address,
                    input_addr=addresses.inputs[0],
                    output_addr=addresses.outputs[0],
                    stage_id=int(omni_stage_id),
                    queue_length_getter=scheduler.get_num_unfinished_requests,
                )

            sibling_cores: list[tuple[Any, StageEngineCoreProc, Any]] = []  # (stage_id, core, stream)
            if sibling_stage_kwargs:
                import threading as _threading

                import torch as _torch

                # PHASE 1 -- construct ALL guest cores sequentially on THIS
                # thread, before ANY guest loop starts: vllm's env cache froze
                # at the primary's init, and no engine's CUDA graph capture may
                # interleave with another engine's running loop. Build order ==
                # the parent's wait_for_engine_startup order (same list).
                for sibling_entry in sibling_stage_kwargs:
                    sib = dict(sibling_entry)
                    sib_stage_id = sib.pop("omni_stage_id", None)
                    sib_replica_id = sib.pop("omni_replica_id", 0)
                    sib_dp_rank = sib.pop("dp_rank", 0)
                    maybe_apply_audex_cfg_patches(sib.get("vllm_config"))
                    logger.info(
                        "[colocate] building sibling stage %s core inside stage %s process",
                        sib_stage_id,
                        omni_stage_id,
                    )
                    # Each sibling lives on its OWN CUDA stream so its kernels
                    # can overlap the others' instead of queueing behind them
                    # on the default stream. Construction happens inside the
                    # stream scope too: graph captures then replay there, and
                    # the stream doubles as the engine's identity for the
                    # stream-keyed MoE workspace arenas.
                    sib_stream = _torch.cuda.Stream()
                    with _torch.cuda.stream(sib_stream):
                        sib_core = StageEngineCoreProc(
                            engine_index=sib_dp_rank,
                            **sib,
                        )
                    sibling_cores.append((sib_stage_id, sib_core, sib_stream))
                    logger.info(
                        "[colocate] sibling stage %s core built (replica %s)",
                        sib_stage_id,
                        sib_replica_id,
                    )

                # PHASE 2 -- start every guest's busy loop thread.
                def _make_sibling_loop(loop_stage_id: Any, loop_core: StageEngineCoreProc, loop_stream: Any):
                    def _sibling_loop() -> None:
                        # Presence-over-absence: this function's EXIT is the
                        # death signal. A silent return would leave the client
                        # waiting forever, so both paths log and notify.
                        try:
                            with _torch.cuda.stream(loop_stream):
                                loop_core.run_busy_loop()
                            logger.warning(
                                "[colocate] sibling stage %s busy loop RETURNED (clean shutdown)",
                                loop_stage_id,
                            )
                        except SystemExit:
                            logger.warning("[colocate] sibling stage %s busy loop exited via SystemExit", loop_stage_id)
                        except Exception:
                            logger.exception(
                                "[colocate] sibling stage %s busy loop DIED; notifying its client",
                                loop_stage_id,
                            )
                            with contextlib.suppress(Exception):
                                loop_core._send_engine_dead()

                    return _sibling_loop

                for sib_stage_id, sib_core, sib_stream in sibling_cores:
                    _threading.Thread(
                        target=_make_sibling_loop(sib_stage_id, sib_core, sib_stream),
                        name=f"StageEngineCore_colocated_stage{sib_stage_id}",
                        daemon=True,
                    ).start()

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                for _, guest_core, _ in sibling_cores:
                    guest_core.shutdown_state = EngineShutdownState.REQUESTED
                    with contextlib.suppress(Exception):
                        guest_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            if host_stream is not None:
                import torch as _torch_host_loop

                with _torch_host_loop.cuda.stream(host_stream):
                    engine_core.run_busy_loop()
            else:
                engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if coord_client is not None:
                with contextlib.suppress(RuntimeError):
                    coord_client.close()
            if engine_core is not None:
                engine_core.shutdown()
