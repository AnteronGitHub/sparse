from .task_deployer_base import TaskDeployerBase

# Select task deployer implementation based on the Python compiler version
import sys

if sys.version_info >= (3, 8, 10):
    from .task_deployer_latest import TaskDeployerLatest as TaskDeployer
else:
    from .task_deployer_legacy import TaskDeployerLegacy as TaskDeployer

__all__ = ["TaskDeployerBase", "TaskDeployer"]
