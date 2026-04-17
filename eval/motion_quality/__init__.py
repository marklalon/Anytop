"""Distribution-based motion quality evaluation — no trained model required."""

from .scorer import DistributionMotionQualityScorer, DistributionEvalReport

__all__ = ["DistributionMotionQualityScorer", "DistributionEvalReport"]
