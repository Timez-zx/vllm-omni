import gc
import os

import torch
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import OmniACK, OmniSleepTask, OmniWakeTask
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.memory_utils import request_memory_tolerant
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class GPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    """GPU worker for autoregressive omni model stages.

    Extends the base GPUWorker to initialize and manage autoregressive
    model runners for text generation stages (e.g., thinker stages).
    """

    @instrument(span_name="Init device")
    def init_device(self):
        if self.device_config.device_type in ("cuda", "musa"):
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size

            # Publish the logical-to-physical mapping for topology queries
            # such as NIC affinity and P2P checks (upstream PR #45026).
            assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
                assert self.local_rank < len(assigned_physical_gpu_ids), (
                    f"local_rank {self.local_rank} is out of bounds for "
                    f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
                )
                if parallel_config.distributed_executor_backend not in ("ray", "external_launcher"):
                    assert self.parallel_config.local_world_size <= len(assigned_physical_gpu_ids), (
                        f"local_world_size ({self.parallel_config.local_world_size}) "
                        "exceeds assigned_physical_gpu_ids count "
                        f"({len(assigned_physical_gpu_ids)})"
                    )
            else:
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of "
                    f"bounds for {torch.accelerator.device_count()} devices."
                )

            visible_device_index = current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            self.device = current_omni_platform.get_torch_device(visible_device_index)
            torch.accelerator.set_device_index(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.accelerator.empty_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory_tolerant(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug("worker requested memory: %sGiB", format_gib(self.requested_memory))
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        if self.use_v2_model_runner:
            # OMNI: v2 model runner does not yet include omni hooks.
            logger.warning("OMNI GPUARWorker forces v1 model runner for omni hooks.")
            self.use_v2_model_runner = False

        # Construct the model runner
        self.model_runner = GPUARModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def handle_sleep_task(self, task: OmniSleepTask | dict) -> OmniACK:
        """
        Explicitly handle sleep commands.
        Calls the implementation in the base class OmniGPUWorkerBase.
        """
        logger.debug(f"[AR Worker {self.rank}] Resolving handle_sleep_task dispatch")
        if isinstance(task, dict):
            task = OmniSleepTask(**task)
        return super().handle_sleep_task(task)

    def handle_wake_task(self, task: OmniWakeTask | dict) -> OmniACK:
        """
        Explicitly handle wake-up commands.
        Calls the implementation in the base class OmniGPUWorkerBase.
        """
        logger.debug(f"[AR Worker {self.rank}] Resolving handle_wake_task dispatch")
        if isinstance(task, dict):
            task = OmniWakeTask(**task)
        return super().handle_wake_task(task)

    def start_pd_cache_sync(self, metadata) -> bool:
        """Start a D-side NIXL import without entering execute_model()."""
        from vllm_omni.engine.nixl_delta_push_connector import (
            NixlDeltaPushConnectorWorker,
        )

        connector = get_kv_transfer_group()
        worker = getattr(connector, "connector_worker", None)
        if not isinstance(worker, NixlDeltaPushConnectorWorker):
            raise RuntimeError("Direct P/D cache sync requires NixlDeltaPushConnectorWorker")
        worker.start_direct_cache_sync(metadata)
        return True

    def poll_pd_cache_sync(self, wait_for_completion: bool = True) -> set[str]:
        """Poll cache-only imports without consuming inference completions."""
        from vllm_omni.engine.nixl_delta_push_connector import (
            NixlDeltaPushConnectorWorker,
        )

        connector = get_kv_transfer_group()
        worker = getattr(connector, "connector_worker", None)
        if not isinstance(worker, NixlDeltaPushConnectorWorker):
            raise RuntimeError("Direct P/D cache sync requires NixlDeltaPushConnectorWorker")
        return worker.poll_direct_cache_sync(wait_for_completion=wait_for_completion)

    def publish_pd_finished_blocks(self, metadata) -> bool:
        """Wake P's NIXL writer without waiting for another model batch."""
        from vllm_omni.engine.nixl_delta_push_connector import (
            NixlDeltaPushConnectorWorker,
        )

        connector = get_kv_transfer_group()
        worker = getattr(connector, "connector_worker", None)
        if not isinstance(worker, NixlDeltaPushConnectorWorker):
            raise RuntimeError(
                "Immediate P/D publication requires "
                "NixlDeltaPushConnectorWorker"
            )
        worker.start_load_kv(metadata)
        return True

    @torch.inference_mode()
    def preencode_minicpmo45_vision(
        self,
        jobs: list[dict[str, object]],
    ) -> dict[str, object]:
        """Run MiniCPM arrival-side vision encoding without an LLM request."""
        model = getattr(self.model_runner, "model", None)
        preencode = getattr(model, "preencode_duplex_vision", None)
        if not callable(preencode):
            return {"supported": False, "encoded_frames": 0}
        return self._run_minicpmo_encoder("vision", preencode, jobs)

    @torch.inference_mode()
    def preencode_minicpmo45_audio(
        self,
        jobs: list[dict[str, object]],
    ) -> dict[str, object]:
        """Run MiniCPM arrival-side audio encoding without an LLM request."""
        model = getattr(self.model_runner, "model", None)
        preencode = getattr(model, "preencode_duplex_audio", None)
        if not callable(preencode):
            return {
                "supported": False,
                "encoded_jobs": 0,
                "job_results": {},
            }
        return self._run_minicpmo_encoder("audio", preencode, jobs)

    def _run_minicpmo_encoder(self, modality, preencode, jobs):
        """One private CUDA stream per ordered encoder executor.

        A background CPU thread alone does not select a new CUDA stream.
        Fence only this encoder's work before acknowledging cache readiness;
        never synchronize the whole device (which would also wait for P).
        """
        raw_device = os.environ.get(f"MINICPMO45_{modality.upper()}_ENCODER_DEVICE", "").strip()
        device = torch.device(f"cuda:{raw_device}" if raw_device.isdigit() else raw_device or self.device)
        if device.type != "cuda":
            return preencode(jobs)
        attr = f"_minicpmo_{modality}_cuda_stream"
        stream = getattr(self, attr, None)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            # Loaded weights precede sidecar serving. Make their default-stream
            # initialization visible on the new stream exactly once.
            stream.wait_stream(torch.cuda.default_stream(device))
            setattr(self, attr, stream)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            try:
                return preencode(jobs)
            finally:
                # Also fence failed work before this executor accepts a retry.
                ready = torch.cuda.Event()
                ready.record(stream)
                ready.synchronize()
