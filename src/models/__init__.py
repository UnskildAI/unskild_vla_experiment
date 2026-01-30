"""Model components: VLM, diffusion, vision, language, action."""

from src.core.registry import ModelRegistry
from src.models.vlm.base import BaseVLM
from src.models.vlm.paligemma import PaliGemmaVLM
from src.models.vlm.llava import LLaVAVLM
from src.models.diffusion.base import BaseDiffusionModel
from src.models.diffusion.flow_matching import FlowMatchingModel
from src.models.action.base import BaseActionHead
from src.models.action.policies import MLPActionHead, DiffusionActionHead, VisionActionWrapper

ModelRegistry.register("paligemma")(PaliGemmaVLM)
ModelRegistry.register("llava")(LLaVAVLM)
ModelRegistry.register("flow_matching")(FlowMatchingModel)
ModelRegistry.register("mlp_action")(MLPActionHead)
ModelRegistry.register("diffusion_action")(DiffusionActionHead)
ModelRegistry.register("vision_action")(VisionActionWrapper)

__all__ = [
    "BaseVLM",
    "BaseDiffusionModel",
    "BaseActionHead",
    "PaliGemmaVLM",
    "LLaVAVLM",
    "FlowMatchingModel",
    "MLPActionHead",
    "DiffusionActionHead",
]
