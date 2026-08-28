# SPDX-License-Identifier: Apache-2.0
"""Three-stage DuplexOmni pipeline.

The checkpoint architecture is identical to Qwen3-Omni, while the Talker
history contract is model-specific.  This pipeline is selected explicitly by
``pipeline: duplexomni`` in the deploy YAML so ordinary Qwen models retain
their original behavior.
"""

from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.duplexomni"

DUPLEXOMNI_PIPELINE = PipelineConfig(
    model_type="duplexomni",
    default_deploy_config_name="duplexomni_3gpu.yaml",
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    endpoint_restrictions=(
        EndpointRestriction(
            OmniServingCapability.COMPLETIONS,
            "DuplexOmni requires ChatML plus audio/video content; use /v1/chat/completions.",
        ),
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.thinker2talker_full_payload",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.thinker2talker_token_only",
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            sampling_constraints={"detokenize": False, "stop_token_ids": [2150]},
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="thinker_config",
            engine_output_type="audio",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
