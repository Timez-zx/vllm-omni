"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import contextlib
import os
import queue
import signal
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from typing import Any, Callable

import vllm.v1.engine.core as _vllm_engine_core_module
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import (
    decorate_logs,
    set_process_title,
)
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequestType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    EngineZmqAddresses,
    SignalCallback,
)

from vllm_omni.distributed.omni_coordinator import create_stage_coord_client
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.stage_init_utils import (
    maybe_apply_audex_cfg_patches,
    set_death_signal,
)

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


class _StageInputQueue(queue.Queue[tuple[EngineCoreRequestType, Any]]):
    """FIFO for data requests with a direct auxiliary-sidecar dispatch."""

    def __init__(
        self,
        auxiliary_dispatch: Callable[[tuple[EngineCoreRequestType, Any]], None]
        | None = None,
    ) -> None:
        super().__init__()
        self._auxiliary_dispatch = auxiliary_dispatch

    @staticmethod
    def _is_auxiliary_vision_rpc(item: tuple[EngineCoreRequestType, Any]) -> bool:
        request_type, request = item
        if request_type != EngineCoreRequestType.UTILITY:
            return False
        try:
            _client_idx, _call_id, method_name, args = request
            return (
                method_name == "collective_rpc"
                and bool(args)
                and args[0] == "preencode_minicpmo45_vision"
            )
        except (TypeError, ValueError, IndexError):
            return False

    def put(
        self,
        item: tuple[EngineCoreRequestType, Any],
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        dispatch = self._auxiliary_dispatch
        if dispatch is not None and self._is_auxiliary_vision_rpc(item):
            # The RPC itself immediately returns a Future backed by the
            # dedicated sidecar executor. Dispatching it from the ZMQ IO
            # thread avoids waiting for Core's current Thinker batch to end.
            dispatch(item)
            return
        super().put(item, block=block, timeout=timeout)


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Base init starts the ZMQ input thread but the Core busy loop does not
        # start until this constructor returns. The IO thread dereferences
        # ``self.input_queue`` on every put, so replacing the still-unconsumed
        # queue here safely gives the auxiliary encoder RPC priority over an
        # accumulated ADD backlog. It never interrupts an executing batch.
        old_queue = self.input_queue
        priority_queue = _StageInputQueue(
            lambda item: self._handle_client_request(*item),
        )
        # Publish the replacement before draining the old queue.  The input
        # thread resolves ``self.input_queue`` for every message, so this order
        # prevents a request from landing in the old queue after the drain.
        self.input_queue = priority_queue
        while True:
            try:
                priority_queue.put_nowait(old_queue.get_nowait())
            except queue.Empty:
                break

    def preprocess_add_request(self, request: OmniEngineCoreRequest) -> tuple[Any, int]:
        """Preserve omni payloads when vLLM builds its scheduler request."""
        scheduler_request, current_wave = super().preprocess_add_request(request)
        scheduler_request.additional_information = request.additional_information
        scheduler_request.external_req_id = getattr(request, "external_req_id", request.request_id)
        return scheduler_request, current_wave

    def _put_priority_output(self, output: tuple[int, EngineCoreOutputs]) -> None:
        """Expose latency-critical sidecar readiness ahead of data outputs.

        The auxiliary vision encoder runs on its own GPU/thread.  Its admission
        reply is detached from the formal append, but exposing the reply ahead
        of data outputs still bounds control-future lifetime and keeps the
        arrival diagnostic meaningful.  Keep the shared output socket and put
        this tiny reply at the head of its thread-safe queue.
        """
        output_queue = self.output_queue
        with output_queue.not_empty:
            output_queue.queue.appendleft(output)
            output_queue.unfinished_tasks += 1
            output_queue.not_empty.notify()

    def _handle_client_request(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
    ) -> None:
        """Acknowledge auxiliary vision work when it enters the sidecar queue.

        The formal Thinker request depends on the worker-local embedding cache,
        not on this control-plane reply. Waiting to acknowledge until GPU work
        completed made the shared Core output path throttle an otherwise
        independent encoder GPU. The sidecar remains single-threaded and the
        cache/tombstone protocol preserves data readiness and correctness.
        """
        if request_type != EngineCoreRequestType.UTILITY:
            return super()._handle_client_request(request_type, request)

        client_idx, call_id, method_name, args = request
        rpc_method = args[0] if method_name == "collective_rpc" and args else None
        if rpc_method != "preencode_minicpmo45_vision":
            return super()._handle_client_request(request_type, request)
        if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
            return

        output = UtilityOutput(call_id)
        enqueue_output = lambda out: self._put_priority_output(
            (client_idx, EngineCoreOutputs(utility_output=out))
        )
        try:
            method = getattr(self, method_name)
            converted_args = self._convert_msgspec_args(method, args)
            result = method(*converted_args)
            if isinstance(result, Future):
                result.add_done_callback(self._consume_auxiliary_vision_result)
                encoded_frames = self._auxiliary_vision_frame_count(args)
                output.result = UtilityResult(
                    [
                        {
                            "supported": True,
                            "accepted": True,
                            # Retain the existing response field so older API
                            # processes interpret queue admission as success.
                            "encoded_frames": encoded_frames,
                        }
                    ]
                )
            else:
                output.result = UtilityResult(result)
        except Exception as exc:
            logger.exception("Invocation of %s method failed", method_name)
            output.failure_message = f"Call to {method_name} method failed: {str(exc)}"
        enqueue_output(output)

    @staticmethod
    def _auxiliary_vision_frame_count(args: Any) -> int:
        try:
            jobs = args[2][0]
            return sum(
                len(job.get("video_frames", ()))
                for job in jobs
                if isinstance(job, dict)
            )
        except (TypeError, ValueError, IndexError):
            return 0

    @staticmethod
    def _consume_auxiliary_vision_result(future: Future[Any]) -> None:
        """Observe background failures without delaying the admission ACK."""
        try:
            future.result()
        except CancelledError:
            return
        except Exception:
            logger.exception("MiniCPM-o auxiliary vision preencode failed")

    def collective_rpc(
        self,
        method: Any,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Run only the auxiliary-GPU vision sidecar outside Core's loop."""
        if (
            method == "preencode_minicpmo45_vision"
            and os.environ.get("MINICPMO45_VISION_ENCODER_DEVICE", "").strip()
        ):
            executor = getattr(self, "_vision_preencode_executor", None)
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
        executor = getattr(self, "_vision_preencode_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._vision_preencode_executor = None
        super().shutdown()

    @staticmethod
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

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

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
