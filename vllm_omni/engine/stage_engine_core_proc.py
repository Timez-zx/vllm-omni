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
from contextlib import ExitStack
from typing import Any

import msgspec
import vllm.v1.engine.core as _vllm_engine_core_module
import zmq
from vllm.logger import init_logger
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
_DIAG_STAGE = os.environ.get("VLLM_OMNI_DIAG_STAGE")


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


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
            TensorIpcSender(output_tensor_queue) if output_tensor_queue is not None else None
        )
        super().__init__(*args, **kwargs)

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

                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
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
                        if _LOG_INGRESS_DIAG and (_DIAG_STAGE is None or str(stage_id) == _DIAG_STAGE):
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
        ingress_diag = _LOG_INGRESS_DIAG and (_DIAG_STAGE is None or str(stage_id) == _DIAG_STAGE)
        ingress_start = time.monotonic() if ingress_diag else 0.0
        prepare_lineage = getattr(self.scheduler, "prepare_kv_lineage_request", None)
        if prepare_lineage is not None:
            prepare_lineage(request)
        lineage_done = time.monotonic() if ingress_diag else 0.0
        # D already receives the complete prompt KV from P. Its lightweight
        # mm_features exist only so Qwen can reconstruct M-RoPE positions.
        # Bypass upstream's D-local media-cache lookup, then restore the
        # position metadata on the scheduler request for the model runner.
        pd_mm_features = None
        if getattr(request, "pd_prefill_payload", None) is not None and request.mm_features:
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
