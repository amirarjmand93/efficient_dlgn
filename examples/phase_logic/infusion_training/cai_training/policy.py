"""Continuous candidate priors for CompactLogic convolution and dense stages."""

from types import MethodType

import torch

from ...compact import CandidateLogicLayer


class ContinuousCandidatePolicy:
    """Change future candidate draws; already selected gates are left alone.

    Alpha zero calls the original uniform resampler. Alpha one applies the
    configured CAI target draw mixture.
    The uniform component can itself draw unary or constant operations.
    """

    CONV_PREFERENCE = 0.25
    DENSE_TARGETS = ((0.50, 0.10, 0.30), (0.45, 0.10, 0.25))
    STRONG_UNARY_TARGETS = ((0.70, 0.15, 0.05), (0.60, 0.15, 0.05))
    ULTRA_UNARY_TARGETS = ((0.90, 0.07, 0.01), (0.86, 0.10, 0.01))

    def __init__(self, model, *, infusion_profile="default"):
        if infusion_profile not in {"default", "strong_unary", "ultra_unary"}:
            raise ValueError(f"Unknown CAI infusion profile: {infusion_profile!r}")
        self.infusion_profile = infusion_profile
        self.dense_targets = {
            "default": self.DENSE_TARGETS,
            "strong_unary": self.STRONG_UNARY_TARGETS,
            "ultra_unary": self.ULTRA_UNARY_TARGETS,
        }[infusion_profile]
        layers = [(name, layer) for name, layer in model.named_modules()
                  if isinstance(layer, CandidateLogicLayer)]
        self.convs = [layer for _, layer in layers if layer.is_convolution]
        self.dense = [layer for _, layer in layers if not layer.is_convolution]
        if not layers:
            raise ValueError("CAI requires at least one CompactLogic candidate layer.")
        # Class-routed heads can contain several candidate engines per dense stage.
        self.dense_stages = {}
        for name, layer in layers:
            if not layer.is_convolution:
                stage_name = name.split(".routes.", 1)[0]
                self.dense_stages.setdefault(stage_name, []).append(layer)
        self.biased_convs = self.convs[:-1]
        self._original_resample = {}
        for layer in self.biased_convs:
            self._original_resample[layer] = layer.resample
            layer.resample = MethodType(self._resample_conv, layer)
        self.alpha = 0.0
        self.set_alpha(0.0)

    def set_alpha(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("CAI alpha must be in [0, 1].")
        self.alpha = float(alpha)
        for layer in self.convs:
            layer.resample_preferred_ops = ()
            layer.cai_preferred_fraction = (
                self.CONV_PREFERENCE * self.alpha if layer in self.biased_convs else 0.0
            )
        for index, engines in enumerate(self.dense_stages.values()):
            target = self.dense_targets[min(index, len(self.dense_targets) - 1)]
            for layer in engines:
                layer.resample_preferred_ops = ()
                layer.resample_efficient_dense = (
                    tuple(self.alpha * fraction for fraction in target)
                    if self.alpha > 0.0 else False
                )

    @torch.no_grad()
    def _resample_conv(self, layer, rows, optimizer):
        fraction = layer.cai_preferred_fraction
        if fraction == 0.0:
            return self._original_resample[layer](rows, optimizer)
        winners = layer.weights.argmax(-1)
        weak, strong = layer.impl.resample_connections(
            rows, threshold_high=0.95, threshold_low=0.4,
            keep_strong_weights=False,
        )
        slots = (weak | strong)[:, None].expand_as(layer.weights).clone()
        strong_rows = strong.nonzero(as_tuple=True)[0]
        slots[strong_rows, winners[strong_rows]] = False
        codes = layer.impl.gate_sequence[slots]
        if codes.numel():
            biased = torch.rand(codes.shape, device=codes.device) < fraction
            pool = codes.new_tensor((1, 7))  # AND / OR, as in aggressive ELAR.
            choices = torch.randint(2, codes.shape, device=codes.device)
            layer.impl.gate_sequence[slots] = torch.where(biased, pool[choices], codes)
        changed = (weak | strong)[:, None].expand_as(layer.weights)
        state = optimizer.state.get(layer.weights, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in state:
                state[name].masked_fill_(changed, 0)
        return int(weak.sum()), int(strong.sum())

    def draw_mixtures(self, alpha=None) -> dict:
        strength = self.alpha if alpha is None else float(alpha)
        mixtures = {}
        if self.biased_convs:
            mixtures[f"conv1_to_{len(self.biased_convs)}"] = {
                "uniform": 1.0 - self.CONV_PREFERENCE * strength,
                "and_or": self.CONV_PREFERENCE * strength,
            }
        if self.convs:
            mixtures[f"conv{len(self.convs)}"] = {"uniform": 1.0}
        for index in range(len(self.dense_stages)):
            target = self.dense_targets[min(index, len(self.dense_targets) - 1)]
            mixtures[f"dense{index + 1}"] = {
                "uniform": 1.0 - sum(target) * strength,
                "identity": target[0] * strength,
                "inversion": target[1] * strength,
                "constant": target[2] * strength,
            }
        return mixtures
