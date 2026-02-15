"""
A/B Testing Framework (Phase 6 - Data & Analytics)

Implements A/B testing and experimentation:
- Experiment creation and management
- User assignment to variants
- Statistical analysis
- Results tracking
"""

import hashlib
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class Variant(BaseModel):
    """Experiment variant."""
    name: str
    weight: float = 50.0  # Percentage weight
    description: Optional[str] = None


class Experiment(BaseModel):
    """A/B test experiment."""
    experiment_id: str
    name: str
    description: Optional[str] = None
    variants: list[Variant]
    status: str = "draft"  # draft, running, paused, completed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    target_sample_size: int = 1000
    metadata: dict = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Results for an experiment."""
    experiment_id: str
    variant_name: str
    participants: int
    conversions: int
    conversion_rate: float
    confidence: float


class ConversionEvent(BaseModel):
    """Conversion event for experiments."""
    experiment_id: str
    variant_name: str
    user_id: str
    converted: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    value: float = 0.0


# In-memory storage
_experiments: dict[str, Experiment] = {}
_user_assignments: dict[str, dict[str, str]] = defaultdict(dict)  # user_id -> {exp_id: variant}
_variant_stats: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
    "participants": 0,
    "conversions": 0,
    "total_value": 0.0,
}))


class ABTestingFramework:
    """A/B testing framework for experiments."""
    
    def __init__(self):
        pass
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        variants: list[dict],
        description: str = None,
        target_sample_size: int = 1000,
    ) -> Experiment:
        """Create a new experiment."""
        variant_objects = [Variant(**v) for v in variants]
        
        # Validate weights sum to 100
        total_weight = sum(v.weight for v in variant_objects)
        if abs(total_weight - 100.0) > 0.01:
            # Normalize weights
            for v in variant_objects:
                v.weight = (v.weight / total_weight) * 100
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variant_objects,
            target_sample_size=target_sample_size,
        )
        
        _experiments[experiment_id] = experiment
        
        logger.info(
            "experiment_created",
            experiment_id=experiment_id,
            variants=[v.name for v in variant_objects],
        )
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        if experiment_id not in _experiments:
            return False
        
        experiment = _experiments[experiment_id]
        experiment.status = "running"
        experiment.started_at = datetime.now(timezone.utc)
        
        logger.info("experiment_started", experiment_id=experiment_id)
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        if experiment_id not in _experiments:
            return False
        
        experiment = _experiments[experiment_id]
        experiment.status = "completed"
        experiment.ended_at = datetime.now(timezone.utc)
        
        logger.info("experiment_stopped", experiment_id=experiment_id)
        return True
    
    def get_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Get the variant assignment for a user."""
        if experiment_id not in _experiments:
            return None
        
        experiment = _experiments[experiment_id]
        
        # Only running experiments assign variants
        if experiment.status != "running":
            return None
        
        # Check existing assignment
        if experiment_id in _user_assignments.get(user_id, {}):
            return _user_assignments[user_id][experiment_id]
        
        # Assign new variant based on hash for consistency
        variant = self._assign_variant(experiment, user_id)
        
        # Store assignment
        _user_assignments[user_id][experiment_id] = variant
        _variant_stats[experiment_id][variant]["participants"] += 1
        
        logger.debug(
            "variant_assigned",
            experiment_id=experiment_id,
            user_id=user_id,
            variant=variant,
        )
        
        return variant
    
    def _assign_variant(self, experiment: Experiment, user_id: str) -> str:
        """Assign a variant to a user using weighted random selection."""
        # Use hash for deterministic assignment
        hash_input = f"{experiment.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 100  # 0-100
        
        # Weighted selection
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant.name
        
        # Fallback to last variant
        return experiment.variants[-1].name
    
    def track_conversion(
        self,
        experiment_id: str,
        user_id: str,
        value: float = 1.0,
    ) -> bool:
        """Track a conversion for an experiment."""
        if experiment_id not in _experiments:
            return False
        
        # Get user's variant
        variant = _user_assignments.get(user_id, {}).get(experiment_id)
        if not variant:
            return False
        
        # Update stats
        _variant_stats[experiment_id][variant]["conversions"] += 1
        _variant_stats[experiment_id][variant]["total_value"] += value
        
        logger.info(
            "conversion_tracked",
            experiment_id=experiment_id,
            user_id=user_id,
            variant=variant,
            value=value,
        )
        
        return True
    
    def get_results(self, experiment_id: str) -> list[ExperimentResult]:
        """Get results for an experiment."""
        if experiment_id not in _experiments:
            return []
        
        experiment = _experiments[experiment_id]
        results = []
        
        for variant in experiment.variants:
            stats = _variant_stats[experiment_id][variant.name]
            participants = stats["participants"]
            conversions = stats["conversions"]
            
            conversion_rate = conversions / participants if participants > 0 else 0.0
            
            # Simple confidence calculation (use proper statistics in production)
            confidence = min(participants / experiment.target_sample_size, 1.0) * 100
            
            results.append(ExperimentResult(
                experiment_id=experiment_id,
                variant_name=variant.name,
                participants=participants,
                conversions=conversions,
                conversion_rate=round(conversion_rate, 4),
                confidence=round(confidence, 2),
            ))
        
        return results
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment details."""
        return _experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[str] = None,
    ) -> list[Experiment]:
        """List all experiments."""
        experiments = list(_experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        return experiments
    
    def get_winner(self, experiment_id: str) -> Optional[str]:
        """Determine the winning variant."""
        results = self.get_results(experiment_id)
        if not results:
            return None
        
        # Find variant with highest conversion rate
        best = max(results, key=lambda r: r.conversion_rate)
        
        # Only declare winner if confidence is high enough
        if best.confidence >= 95 and best.participants >= 100:
            return best.variant_name
        
        return None


# Global instance
ab_testing = ABTestingFramework()


# Pre-configured experiments
def setup_default_experiments():
    """Set up default experiments."""
    # UI experiment
    ab_testing.create_experiment(
        experiment_id="ui_theme_v1",
        name="UI Theme Test",
        description="Test dark vs light theme preference",
        variants=[
            {"name": "dark", "weight": 50},
            {"name": "light", "weight": 50},
        ],
    )
    
    # Pricing experiment
    ab_testing.create_experiment(
        experiment_id="pricing_v1",
        name="Pricing Page Test",
        description="Test different pricing presentations",
        variants=[
            {"name": "monthly", "weight": 33.33},
            {"name": "annual", "weight": 33.33},
            {"name": "both", "weight": 33.34},
        ],
    )
    
    # Feature experiment
    ab_testing.create_experiment(
        experiment_id="onboarding_v1",
        name="Onboarding Flow Test",
        description="Test different onboarding experiences",
        variants=[
            {"name": "minimal", "weight": 50},
            {"name": "guided", "weight": 50},
        ],
    )
