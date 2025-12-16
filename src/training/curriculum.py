"""Curriculum learning strategies for Phase 1.5."""

import random
from typing import List


class MixedCurriculumScheduler:
    """Samples corruption rates from distribution to preserve one-shot accuracy."""

    def __init__(
        self,
        epoch_stages: dict[int, float] | None = None,
        mix_ratio: float = 0.3,
    ):
        """
        Initialize mixed curriculum scheduler.

        Args:
            epoch_stages: {epoch: primary_corruption_rate} mapping for curriculum stages
            mix_ratio: Fraction of batch to sample from lower corruption stages
                      (0.3 = 30% of samples use earlier stages, even in late training)
        """
        # Default 5-stage curriculum from phase1.5.md
        self.epoch_stages = epoch_stages or {
            0: 0.2,   # Stage 1 (epochs 0-5):   20% corruption
            6: 0.5,   # Stage 2 (epochs 6-15):  50% corruption
            16: 0.75, # Stage 3 (epochs 16-25): 75% corruption
            26: 0.9,  # Stage 4 (epochs 26-40): 90% corruption
            41: 1.0,  # Stage 5 (epochs 41+):   100% corruption (full generation)
        }
        self.mix_ratio = mix_ratio

    def get_corruption_rate(
        self,
        epoch: int,
        batch_idx: int,
        batch_size: int = 1,
    ) -> list[float]:
        """
        Sample corruption rate for each sample in batch.

        Args:
            epoch: Current training epoch
            batch_idx: Batch index (unused, for future per-batch scheduling)
            batch_size: Number of samples in batch

        Returns:
            List of corruption rates, one per sample
        """
        # Determine primary rate for current epoch
        primary_rate = self._get_stage_rate(epoch)

        # Get all rates lower than primary
        lower_rates = [
            rate for stage_epoch, rate in self.epoch_stages.items()
            if rate < primary_rate
        ]

        corruption_rates = []
        for _ in range(batch_size):
            # With probability mix_ratio, sample from earlier stages
            if lower_rates and random.random() < self.mix_ratio:
                # Sample uniformly from earlier stages
                rate = random.choice(lower_rates)
            else:
                # Use primary rate for current epoch
                rate = primary_rate

            corruption_rates.append(rate)

        return corruption_rates

    def _get_stage_rate(self, epoch: int) -> float:
        """
        Get primary corruption rate for current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            Corruption rate for this epoch's stage
        """
        # Find the stage for this epoch
        stages = sorted(self.epoch_stages.items())
        for stage_epoch, rate in reversed(stages):
            if epoch >= stage_epoch:
                return rate

        # Fallback: first stage rate
        return stages[0][1]

    def get_current_stage_info(self, epoch: int) -> dict:
        """
        Get information about current curriculum stage.

        Args:
            epoch: Current epoch

        Returns:
            Dict with stage info: {
                'stage_number': int,
                'primary_rate': float,
                'lower_rates': List[float],
                'mix_ratio': float,
            }
        """
        primary_rate = self._get_stage_rate(epoch)

        # Determine stage number (1-indexed)
        stage_number = 1
        for i, (stage_epoch, rate) in enumerate(sorted(self.epoch_stages.items()), start=1):
            if epoch >= stage_epoch:
                stage_number = i

        lower_rates = [
            rate for rate in self.epoch_stages.values()
            if rate < primary_rate
        ]

        return {
            'stage_number': stage_number,
            'primary_rate': primary_rate,
            'lower_rates': sorted(lower_rates),
            'mix_ratio': self.mix_ratio,
            'effective_rates': self._compute_effective_distribution(epoch),
        }

    def _compute_effective_distribution(self, epoch: int) -> dict[float, float]:
        """
        Compute effective probability distribution over corruption rates.

        Args:
            epoch: Current epoch

        Returns:
            Dict mapping corruption_rate → probability
        """
        primary_rate = self._get_stage_rate(epoch)
        lower_rates = [
            rate for rate in self.epoch_stages.values()
            if rate < primary_rate
        ]

        if not lower_rates:
            # Only primary rate available
            return {primary_rate: 1.0}

        # Primary rate gets (1 - mix_ratio) probability
        # Lower rates split mix_ratio equally
        dist = {primary_rate: 1.0 - self.mix_ratio}

        prob_per_lower = self.mix_ratio / len(lower_rates)
        for rate in lower_rates:
            dist[rate] = prob_per_lower

        return dist
