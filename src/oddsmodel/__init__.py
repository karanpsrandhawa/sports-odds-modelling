"""Sports odds modelling: de-vigging, market consensus and mispricing detection."""
from .consensus import Flag, find_mispricings
from .probability import fair_probabilities, overround

__all__ = ["Flag", "find_mispricings", "fair_probabilities", "overround"]
__version__ = "0.1.0"
