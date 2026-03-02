"""Validation loss computation during training.

Runs a fixed number of steps on a validation dataset at configured intervals,
using the strategy's standard training_step for reproducibility.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.utils.data
    from trainer.arch.base import ModelStrategy, ModelComponents

logger = logging.getLogger(__name__)


class ValidationRunner:
    """Runs validation loop and returns metrics.

    Operates entirely under ``torch.no_grad()`` and temporarily switches the
    model to eval mode via the strategy's ``on_before_sampling`` /
    ``on_after_sampling`` hooks, which mirrors what sample generation does.
    """

    def __init__(
        self,
        strategy: ModelStrategy,
        components: ModelComponents,
        dataloader: torch.utils.data.DataLoader,
        num_steps: int = 10,
    ):
        self.strategy = strategy
        self.components = components
        self.dataloader = dataloader
        self.num_steps = num_steps
        self._iter = None

    def _get_batch(self) -> dict:
        """Get next batch, cycling the dataloader infinitely."""
        if self._iter is None:
            self._iter = iter(self.dataloader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            return next(self._iter)

    @torch.no_grad()
    def run(self, step: int) -> dict[str, float]:
        """Run validation and return metrics.

        Args:
            step: Current training step (used for logging only).

        Returns:
            Dict with at least ``"val_loss"`` key.
        """
        self.strategy.on_before_sampling(self.components)  # switch to eval mode

        total_loss = 0.0
        count = 0

        try:
            for _ in range(self.num_steps):
                batch = self._get_batch()
                output = self.strategy.training_step(self.components, batch, step)
                total_loss += output.loss.detach().item()
                count += 1
        finally:
            self.strategy.on_after_sampling(self.components)  # back to train mode

        avg_loss = total_loss / max(count, 1)
        logger.info(
            "Validation at step %d: val_loss=%.6f (%d steps)",
            step, avg_loss, count,
        )

        return {"val_loss": avg_loss}
