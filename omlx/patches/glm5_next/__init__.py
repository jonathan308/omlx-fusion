"""GLM-5.3-Flash contract and conversion-planning support.

This package intentionally does not register ``glm5_next`` as an mlx-lm
architecture.  The checkpoint uses math which oMLX does not yet implement;
pretending it is the older ``glm_moe_dsa`` model would silently produce
incorrect results.
"""

from .contract import (
    OFFICIAL_REVISION,
    Glm5NextContractError,
    Glm5NextSourceContract,
    validate_config,
    validate_source_contract,
    validate_weight_index,
)
from .convert import (
    Glm5NextConversionPlan,
    Glm5NextUnsupportedMathError,
    build_conversion_plan,
    convert_glm53_flash,
)

__all__ = [
    "OFFICIAL_REVISION",
    "Glm5NextContractError",
    "Glm5NextConversionPlan",
    "Glm5NextSourceContract",
    "Glm5NextUnsupportedMathError",
    "build_conversion_plan",
    "convert_glm53_flash",
    "validate_config",
    "validate_source_contract",
    "validate_weight_index",
]
