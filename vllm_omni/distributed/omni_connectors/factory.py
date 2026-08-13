# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import Any

from .utils.logging import get_connector_logger

try:
    from .connectors.base import OmniConnectorBase
    from .utils.config import ConnectorSpec
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from omni_connectors.connectors.base import OmniConnectorBase
    from omni_connectors.utils.config import ConnectorSpec

logger = get_connector_logger(__name__)


class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""

    _registry: dict[str, Callable[[dict[str, Any]], OmniConnectorBase]] = {}

    @classmethod
    def register_connector(cls, name: str, constructor: Callable[[dict[str, Any]], OmniConnectorBase]) -> None:
        """Register a connector constructor."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = constructor
        logger.debug(f"Registered connector: {name}")

    @classmethod
    def create_connector(cls, spec: ConnectorSpec) -> OmniConnectorBase:
        """Create a connector from specification."""
        if spec.name not in cls._registry:
            raise ValueError(f"Unknown connector: {spec.name}. Available: {list(cls._registry.keys())}")

        constructor = cls._registry[spec.name]
        try:
            connector = constructor(spec.extra)
            logger.info(f"Created connector: {spec.name}")
            return connector
        except Exception as e:
            logger.error(f"Failed to create connector {spec.name}: {e}")
            raise ValueError(f"Failed to create connector {spec.name}: {e}")

    @classmethod
    def list_registered_connectors(cls) -> list[str]:
        """List all registered connector names."""
        return list(cls._registry.keys())


# Register built-in connectors with lazy imports
def _create_mooncake_store_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_store_connector import MooncakeStoreConnector
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_store_connector import MooncakeStoreConnector
    return MooncakeStoreConnector(config)


def _create_shm_connector(config: dict[str, Any]) -> OmniConnectorBase:
    # [Tick engine WP4] VLLM_OMNI_TEMPORAL_MAILBOX=1 substitutes the
    # persistent two-slot mailbox wherever the deploy yaml says
    # SharedMemoryConnector -- experiment arms keep ONE shared yaml and
    # differ only by env. The mailbox inherits the legacy path as its
    # universal fallback, so behavior is a superset.
    from vllm_omni.core.sched.temporal_pacing import live_env_on as _live_env_on
    if _live_env_on("VLLM_OMNI_TEMPORAL_MAILBOX"):  # live-vllm: default ON
        try:
            from .connectors.tick_mailbox_connector import TickMailboxConnector

            return TickMailboxConnector(config)
        except Exception as e:
            logger.warning(f"TickMailboxConnector unavailable ({e}); using SharedMemoryConnector")
    try:
        from .connectors.shm_connector import SharedMemoryConnector
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.shm_connector import SharedMemoryConnector
    return SharedMemoryConnector(config)


def _create_yuanrong_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.yuanrong_connector import YuanrongConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.yuanrong_connector import YuanrongConnector
    return YuanrongConnector(config)


def _create_yuanrong_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from vllm_omni.platforms.npu.omni_connectors import YuanrongTransferEngineConnector
    except ImportError as exc:
        raise ImportError(
            "YuanrongTransferEngineConnector is only available in the NPU platform "
            "environment. Install the Ascend/Yuanrong runtime dependencies before "
            "using this connector."
        ) from exc
    return YuanrongTransferEngineConnector(config)


def _create_mooncake_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
    return MooncakeTransferEngineConnector(config)


def _create_mori_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
    return MoriTransferEngineConnector(config)


# Register connectors
OmniConnectorFactory.register_connector("MooncakeStoreConnector", _create_mooncake_store_connector)
OmniConnectorFactory.register_connector("MooncakeTransferEngineConnector", _create_mooncake_transfer_engine_connector)
OmniConnectorFactory.register_connector("SharedMemoryConnector", _create_shm_connector)


def _create_coloc_inproc_connector(config: dict[str, Any]) -> OmniConnectorBase:
    from .connectors.coloc_inproc_connector import ColocInProcConnector

    return ColocInProcConnector(config)


OmniConnectorFactory.register_connector("ColocInProcConnector", _create_coloc_inproc_connector)
OmniConnectorFactory.register_connector("YuanrongConnector", _create_yuanrong_connector)
OmniConnectorFactory.register_connector("YuanrongTransferEngineConnector", _create_yuanrong_transfer_engine_connector)
OmniConnectorFactory.register_connector("MoriTransferEngineConnector", _create_mori_transfer_engine_connector)
# Backward-compatible aliases – will be removed in the future
OmniConnectorFactory.register_connector("MooncakeConnector", _create_mooncake_store_connector)
