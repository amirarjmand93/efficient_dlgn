"""Opt-in, cross-entropy-probed rewiring of CIFAR-10 logic layers."""

import math
import time

import torch

from torchlogix.layers import ClassRoutedLogicDense, LogicConv2d, LogicDense

from .config import ExperimentConfig


class _SharedDenseRewireAdapter:
    """Expose a plain LogicDense to the existing shared-input probe."""

    shared_input = True
    row_entropy = ClassRoutedLogicDense.row_entropy
    replace_row = ClassRoutedLogicDense.replace_row

    def __init__(self, layer: LogicDense):
        self.layer = layer

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)

    @torch.no_grad()
    def candidate_pairs(self, inputs, output_gradient, row, **kwargs):
        # The first small-model dense layer has redundant input coverage.
        # Never remove the last remaining use of either current input.
        counts = torch.bincount(
            self.connections.indices.flatten(), minlength=self.in_dim
        )
        if bool((counts[self.connections.indices[:, row]] <= 1).any()):
            return self.connections.indices.new_empty((0, 2))
        return ClassRoutedLogicDense.candidate_pairs(
            self, inputs, output_gradient, row, **kwargs
        )


class _CoveragePreservingDenseAdapter(_SharedDenseRewireAdapter):
    """Rewire a shrinking dense layer by swapping two complete input pairs."""

    @torch.no_grad()
    def candidate_swaps(
        self,
        inputs: torch.Tensor,
        output_gradient: torch.Tensor,
        row: int,
        *,
        num_candidates: int,
        allowed_rows: torch.Tensor,
    ) -> torch.Tensor:
        centered = output_gradient - output_gradient.mean()
        scores = torch.mv(inputs.T, centered).abs()
        pair_scores = scores[self.connections.indices].sum(dim=0)
        allowed_rows = allowed_rows.clone()
        allowed_rows[row] = False
        candidates = allowed_rows.nonzero(as_tuple=True)[0]
        if not candidates.numel():
            return candidates
        scores = pair_scores[candidates]
        return candidates[scores.topk(min(num_candidates, candidates.numel())).indices]

    @torch.no_grad()
    def swap_outputs(
        self, inputs: torch.Tensor, row: int, donors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        connections = self.connections.indices
        target_inputs = inputs[:, connections[:, donors]]
        target_weights = self.weight[row].expand(donors.numel(), -1)
        target_outputs = self.parametrization.forward(
            target_inputs, target_weights, True, contraction="n,bn->bn"
        )
        donor_inputs = inputs[:, connections[:, row]].unsqueeze(-1)
        donor_inputs = donor_inputs.expand(-1, -1, donors.numel())
        donor_outputs = self.parametrization.forward(
            donor_inputs, self.weight[donors], True, contraction="n,bn->bn"
        )
        return target_outputs, donor_outputs

    @torch.no_grad()
    def replace_row(
        self,
        row: int,
        donor: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        indices = self.connections.indices
        old_pair = indices[:, row].clone()
        indices[:, row].copy_(indices[:, donor])
        indices[:, donor].copy_(old_pair)
        state = optimizer.state.get(self.weight, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = state.get(name)
            if moment is not None:
                moment[row].zero_()
                moment[donor].zero_()


class _ConvRewireAdapter:
    """Probe one shared spatial leaf pair of a rank-2 convolutional tree."""

    def __init__(self, layer: LogicConv2d, interval: int):
        self.layer = layer
        self.interval = interval
        self.out_dim = layer.num_kernels
        self.weight = layer.tree_weights[0]
        self.parametrization = layer.parametrization
        self.leaf = 0
        indices = layer.connections.indices[0]
        self.kernel_channels = [
            torch.unique(indices[:, row, 0, :, -1])
            for row in range(self.out_dim)
        ]

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)

    @torch.no_grad()
    def row_entropy(self) -> torch.Tensor:
        weights = torch.cat(list(self.layer.tree_weights), dim=0)
        probabilities = torch.sigmoid(
            weights.detach() / self.parametrization.temperature
        ).clamp(1e-7, 1 - 1e-7)
        entropy = -probabilities * probabilities.log()
        entropy -= (1 - probabilities) * torch.log1p(-probabilities)
        return entropy.mean(dim=(0, 2)) / math.log(2)

    @torch.no_grad()
    def candidate_pairs(
        self,
        inputs: torch.Tensor,
        output_gradient: torch.Tensor,
        row: int,
        *,
        feature_pool_size: int,
        num_candidates: int,
        step: int,
    ) -> torch.Tensor:
        del inputs, output_gradient, feature_pool_size
        layer = self.layer
        leaves = layer.tree_weights[0].shape[0]
        self.leaf = (step // self.interval + row) % leaves
        indices = layer.connections.indices[0]
        channels = self.kernel_channels[row]
        height, width = layer.receptive_field_size
        h, w = torch.meshgrid(
            torch.arange(height, device=indices.device),
            torch.arange(width, device=indices.device),
            indexing="ij",
        )
        spatial = torch.stack((h.flatten(), w.flatten()), dim=1)
        positions = torch.cat([
            torch.cat((spatial, channel.expand(spatial.shape[0], 1)), dim=1)
            for channel in channels
        ])
        choices = positions[torch.combinations(
            torch.arange(positions.shape[0], device=indices.device), r=2
        )]
        current = indices[:, row, 0, self.leaf]
        # Preserve the original per-kernel channel counts and distinct leaf
        # positions; otherwise repeated updates can collapse a channel group.
        same_channels = (
            torch.sort(choices[:, :, 2], dim=1).values
            == torch.sort(current[:, 2]).values
        ).all(dim=1)
        other_leaves = indices[:, row, 0]
        other_leaves = torch.cat(
            (other_leaves[:, :self.leaf], other_leaves[:, self.leaf + 1:]),
            dim=1,
        ).reshape(-1, 3)
        conflicts = (
            choices[:, :, None, :] == other_leaves[None, None, :, :]
        ).all(dim=-1).any(dim=-1).any(dim=-1)
        old_pair = ((choices == current).all(dim=2)).all(dim=1)
        old_pair |= ((choices == current.flip(0)).all(dim=2)).all(dim=1)
        choices = choices[same_channels & ~conflicts & ~old_pair]
        if choices.shape[0] == 0:
            return choices
        order = torch.randperm(choices.shape[0], device=choices.device)
        return choices[order[:num_candidates]]

    def _kernel_output(
        self,
        padded: torch.Tensor,
        first_outputs: torch.Tensor,
        row: int,
        pair: torch.Tensor,
        new_weight: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.layer
        old = layer.connections.indices[0][:, row, :, self.leaf]
        offsets = old[0, :, :2] - old[0, 0, :2]
        coords = pair[:, None, :].expand(-1, offsets.shape[0], -1).clone()
        coords[:, :, :2] += offsets
        selected = padded[:, coords[:, :, 2], coords[:, :, 0], coords[:, :, 1]]
        selected = selected.unsqueeze(2).unsqueeze(-1)
        weight = new_weight.reshape(1, 1, -1)
        leaf_output = self.parametrization.forward(
            selected, weight, True, contraction="fc,bcsf->bcsf"
        )[:, 0, :, 0]
        values = first_outputs[:, row].clone()
        values[:, :, self.leaf] = leaf_output
        for level in range(1, layer.tree_depth):
            selected = values[..., layer.connections.indices[level]]
            selected = selected.movedim(-2, 1).unsqueeze(2)
            weights = layer.tree_weights[level][:, row:row + 1]
            values = self.parametrization.forward(
                selected, weights, True, contraction="fc,bcsf->bcsf"
            )[:, 0]
        return values[..., 0].reshape(padded.shape[0], *layer.kernel_positions)

    @torch.no_grad()
    def candidate_outputs(
        self,
        inputs: torch.Tensor,
        row: int,
        pairs: torch.Tensor,
        new_weight: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.layer
        padded = torch.nn.functional.pad(
            inputs, (layer.padding,) * 4, value=0
        ) if layer.padding else inputs
        first_inputs = layer.connections(padded, 0)
        first_outputs = self.parametrization.forward(
            first_inputs, layer.tree_weights[0], True,
            contraction="fc,bcsf->bcsf",
        )
        return torch.stack([
            self._kernel_output(padded, first_outputs, row, pair, new_weight)
            for pair in pairs
        ], dim=1)

    @torch.no_grad()
    def probe_candidate(
        self,
        inputs: torch.Tensor,
        row: int,
        pair: torch.Tensor,
        new_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.candidate_outputs(
            inputs, row, pair.unsqueeze(0), new_weight
        )[:, 0]

    @torch.no_grad()
    def replace_row(
        self,
        row: int,
        pair: torch.Tensor,
        new_weight: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        layer = self.layer
        indices = layer.connections.indices[0]
        old = indices[:, row, :, self.leaf]
        offsets = old[0, :, :2] - old[0, 0, :2]
        old[:, :, :2] = pair[:, None, :2] + offsets
        old[:, :, 2] = pair[:, None, 2]
        weight = layer.tree_weights[0]
        weight[self.leaf, row].copy_(new_weight)
        state = optimizer.state.get(weight, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = state.get(name)
            if moment is not None:
                moment[self.leaf, row].zero_()


class _SingleLayerLossProbedRewirer:
    """Screen weak gates, then verify candidate pairs on a separate train batch."""

    def __init__(
        self,
        model: torch.nn.Sequential,
        config: ExperimentConfig,
        layer_index: int,
        layer,
    ):
        self.layer_index, self.layer = layer_index, layer
        modules = list(model.children())
        self.prefix = torch.nn.Sequential(*modules[:self.layer_index])
        self.suffix = torch.nn.Sequential(*modules[self.layer_index + 1:])
        self.config = config
        self.max_rows = config.rewire_max_rows
        self.previous_entropy = None
        self.last_rewired_step = torch.full(
            (self.layer.out_dim,), -config.rewire_interval,
            device=self.layer.weight.device, dtype=torch.long,
        )
        self.events = 0
        self.attempted_rows = 0
        self.accepted_rows = 0
        self.elapsed_seconds = 0.0

    @staticmethod
    def _row_priority(output: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """Prefer influential gates whose outputs do not help the minibatch loss.

        A negative gradient/output correlation indicates a helpful gate. A
        constant gate has zero correlation but remains eligible if its loss
        gradient is large; a positively correlated gate is potentially harmful.
        """
        centered = output - output.mean(dim=0, keepdim=True)
        effect = (gradient * centered).mean(dim=0)
        scale = (gradient.abs() * centered.abs()).mean(dim=0)
        alignment = (effect / scale.clamp_min(1e-12)).clamp(-1, 1)
        sensitivity = gradient.abs().mean(dim=0)
        return sensitivity * (1 + alignment) / 2

    @torch.no_grad()
    def _discrete_probe_passes(
        self,
        model: torch.nn.Sequential,
        probe_inputs: torch.Tensor,
        probe_targets: torch.Tensor,
        row: int,
        pair: torch.Tensor,
        new_weight: torch.Tensor | None,
    ) -> bool:
        """Reject a small-model change that worsens discrete probe CE."""
        layer = self.layer
        was_training = model.training
        model.eval()
        try:
            before = torch.nn.functional.cross_entropy(
                model(probe_inputs), probe_targets
            )
            if isinstance(layer, _ConvRewireAdapter):
                indices = layer.layer.connections.indices[0]
                old_indices = indices[:, row, :, layer.leaf].clone()
                old_weight = layer.layer.tree_weights[0][layer.leaf, row].clone()
            elif isinstance(layer, _CoveragePreservingDenseAdapter):
                donor = int(pair)
                indices = layer.connections.indices
                old_indices = indices[:, (row, donor)].clone()
                old_weight = None
            else:
                indices = layer.connections.indices
                old_indices = indices[:, row].clone()
                old_weight = layer.weight[row].clone()

            # Trial assignment must not reset real Adam state before acceptance.
            class _NoState:
                state = {}

            try:
                if isinstance(layer, _CoveragePreservingDenseAdapter):
                    layer.replace_row(row, donor, _NoState())
                else:
                    layer.replace_row(row, pair, new_weight, _NoState())
                after = torch.nn.functional.cross_entropy(
                    model(probe_inputs), probe_targets
                )
            finally:
                if isinstance(layer, _ConvRewireAdapter):
                    indices[:, row, :, layer.leaf].copy_(old_indices)
                    layer.layer.tree_weights[0][layer.leaf, row].copy_(old_weight)
                elif isinstance(layer, _CoveragePreservingDenseAdapter):
                    indices[:, (row, donor)] = old_indices
                else:
                    indices[:, row].copy_(old_indices)
                    layer.weight[row].copy_(old_weight)
            return bool(torch.isfinite(after) and after <= before)
        finally:
            model.train(was_training)

    def maybe_rewire(
        self,
        model: torch.nn.Sequential,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
        *,
        probe_inputs: torch.Tensor,
        probe_targets: torch.Tensor,
    ) -> dict | None:
        config = self.config
        if step < config.rewire_warmup or step % config.rewire_interval:
            return None

        layer = self.layer
        if layer.weight.is_cuda:
            torch.cuda.synchronize(layer.weight.device)
        started = time.perf_counter()
        entropy = layer.row_entropy()
        converged = entropy <= config.rewire_entropy_threshold
        if self.previous_entropy is not None:
            stalled = (
                (entropy - self.previous_entropy).abs()
                <= config.rewire_entropy_delta
            )
        else:
            stalled = torch.zeros_like(converged)
        self.previous_entropy = entropy
        eligible = converged | stalled
        eligible &= step - self.last_rewired_step >= config.rewire_interval
        eligible_count = int(eligible.sum().item())
        accepted = 0
        attempted = 0
        search_before = None
        search_after = None
        probe_before = None
        probe_after = None

        if eligible_count and self.max_rows:
            # Cache the expensive encoder/backbone/shared dense representation.
            with torch.no_grad():
                features = self.prefix(inputs)
                current_output = layer(features).detach()
                probe_features = self.prefix(probe_inputs)
                probe_output = layer(probe_features).detach()
                probe_before = float(torch.nn.functional.cross_entropy(
                    self.suffix(probe_output), probe_targets
                ))
            detached_output = current_output.requires_grad_(True)
            with torch.enable_grad():
                reference_loss = torch.nn.functional.cross_entropy(
                    self.suffix(detached_output), targets
                )
                gradient = torch.autograd.grad(reference_loss, detached_output)[0].detach()
            current_output = detached_output.detach()
            search_before = float(reference_loss.detach())
            search_after = search_before
            probe_after = probe_before
            if isinstance(layer, _ConvRewireAdapter):
                priority = self._row_priority(
                    current_output.movedim(1, -1).reshape(-1, layer.out_dim),
                    gradient.movedim(1, -1).reshape(-1, layer.out_dim),
                )
            else:
                priority = self._row_priority(current_output, gradient)
            priority = priority.masked_fill(~eligible, float("-inf"))
            rows = priority.topk(min(self.max_rows, eligible_count)).indices

            with torch.no_grad():
                for row_tensor in rows:
                    row = int(row_tensor)
                    if self.last_rewired_step[row] == step:
                        continue
                    is_swap = isinstance(layer, _CoveragePreservingDenseAdapter)
                    if is_swap:
                        pairs = layer.candidate_swaps(
                            features, gradient[:, row], row,
                            num_candidates=config.rewire_candidates,
                            allowed_rows=(
                                step - self.last_rewired_step
                                >= config.rewire_interval
                            ),
                        )
                    else:
                        pair_kwargs = dict(
                            feature_pool_size=config.rewire_feature_pool,
                            num_candidates=config.rewire_candidates,
                        )
                        if isinstance(layer, _ConvRewireAdapter):
                            pair_kwargs["step"] = step
                        pairs = layer.candidate_pairs(
                            features, gradient[:, row], row, **pair_kwargs
                        )
                    if not pairs.numel():
                        continue
                    attempted += 1
                    new_weight = None
                    if is_swap:
                        candidate_outputs, donor_outputs = layer.swap_outputs(
                            features, row, pairs
                        )
                    else:
                        new_weight = layer.parametrization.init_weights(
                            1, str(layer.weight.device)
                        )[0].to(dtype=layer.weight.dtype)
                    if isinstance(layer, _ConvRewireAdapter):
                        candidate_outputs = layer.candidate_outputs(
                            features, row, pairs, new_weight
                        )
                    elif not is_swap:
                        candidate_inputs = features[:, pairs].permute(0, 2, 1)
                        candidate_weights = new_weight.expand(pairs.shape[0], -1)
                        candidate_outputs = layer.parametrization.forward(
                            candidate_inputs, candidate_weights, True,
                            contraction="n,bn->bn",
                        )
                    gradient_row = gradient[:, row].unsqueeze(1)
                    current_row = current_output[:, row].unsqueeze(1)
                    estimated_change = (
                        gradient_row * (candidate_outputs - current_row)
                    ).sum(dim=tuple(i for i in range(candidate_outputs.ndim) if i != 1))
                    if is_swap:
                        estimated_change += (
                            gradient[:, pairs]
                            * (donor_outputs - current_output[:, pairs])
                        ).sum(dim=0)
                    best = int(estimated_change.argmin())

                    # Require improvement on the search batch and on a separate
                    # training batch that did not generate/rank the candidate.
                    trial_search_output = current_output.clone()
                    trial_search_output[:, row] = candidate_outputs[:, best]
                    if is_swap:
                        donor = int(pairs[best])
                        trial_search_output[:, donor] = donor_outputs[:, best]
                    trial_search_ce = torch.nn.functional.cross_entropy(
                        self.suffix(trial_search_output), targets
                    )
                    if (
                        not torch.isfinite(trial_search_ce)
                        or float(trial_search_ce) >= search_after - config.rewire_min_improvement
                    ):
                        continue
                    if is_swap:
                        candidate_probe_output, donor_probe_output = layer.swap_outputs(
                            probe_features, row, pairs[best:best + 1]
                        )
                        candidate_probe_output = candidate_probe_output[:, 0]
                        donor_probe_output = donor_probe_output[:, 0]
                    elif isinstance(layer, _ConvRewireAdapter):
                        candidate_probe_output = layer.probe_candidate(
                            probe_features, row, pairs[best], new_weight
                        )
                    else:
                        candidate_probe_output = layer.parametrization.forward(
                            probe_features[:, pairs[best]].unsqueeze(-1),
                            new_weight, True, contraction="n,bn->bn",
                        )[:, 0]
                    trial_probe_output = probe_output.clone()
                    trial_probe_output[:, row] = candidate_probe_output
                    if is_swap:
                        trial_probe_output[:, donor] = donor_probe_output
                    trial_probe_ce = torch.nn.functional.cross_entropy(
                        self.suffix(trial_probe_output), probe_targets
                    )
                    if (
                        not torch.isfinite(trial_probe_ce)
                        or float(trial_probe_ce) >= probe_after - config.rewire_min_improvement
                    ):
                        continue
                    if (
                        config.model_variant == "build_model_cifar10_smallmodel"
                        and not self._discrete_probe_passes(
                            model, probe_inputs, probe_targets,
                            row, pairs[best], new_weight,
                        )
                    ):
                        continue
                    if is_swap:
                        layer.replace_row(row, donor, optimizer)
                        self.last_rewired_step[donor] = step
                    else:
                        layer.replace_row(row, pairs[best], new_weight, optimizer)
                    self.last_rewired_step[row] = step
                    current_output = trial_search_output
                    probe_output = trial_probe_output
                    search_after = float(trial_search_ce)
                    probe_after = float(trial_probe_ce)
                    accepted += 1

        if layer.weight.is_cuda:
            torch.cuda.synchronize(layer.weight.device)
        elapsed = time.perf_counter() - started
        self.events += 1
        self.attempted_rows += attempted
        self.accepted_rows += accepted
        self.elapsed_seconds += elapsed
        return {
            "step": step,
            "mean_entropy": float(entropy.mean()),
            "converged_rows": int(converged.sum().item()),
            "stalled_rows": int(stalled.sum().item()),
            "eligible_rows": eligible_count,
            "attempted_rows": attempted,
            "accepted_rows": accepted,
            "search_ce_before": search_before,
            "search_ce_after": search_after,
            "probe_ce_before": probe_before,
            "probe_ce_after": probe_after,
            "elapsed_seconds": elapsed,
        }

    def summary(self) -> dict:
        return {
            "events": self.events,
            "attempted_rows": self.attempted_rows,
            "accepted_rows": self.accepted_rows,
            "elapsed_seconds": self.elapsed_seconds,
        }


class LossProbedRewirer:
    """Keep the class-routed probe unchanged; cover all small-model logic layers."""

    _row_priority = staticmethod(_SingleLayerLossProbedRewirer._row_priority)

    def __init__(self, model: torch.nn.Sequential, config: ExperimentConfig):
        if config.model_variant == "build_model_cifar10_smallmodel":
            targets = [
                (
                    index,
                    _ConvRewireAdapter(module, config.rewire_interval)
                    if isinstance(module, LogicConv2d)
                    else (
                        _CoveragePreservingDenseAdapter(module)
                        if module.in_dim == 2 * module.out_dim
                        else _SharedDenseRewireAdapter(module)
                    ),
                )
                for index, module in enumerate(model)
                if isinstance(module, (LogicConv2d, LogicDense))
            ]
            if sum(isinstance(layer, _ConvRewireAdapter) for _, layer in targets) != 4 or len(targets) != 7:
                raise ValueError("Expected four conv and three dense small-model logic layers.")
        else:
            targets = [
                (index, module)
                for index, module in enumerate(model)
                if isinstance(module, ClassRoutedLogicDense) and module.shared_input
            ]
            if len(targets) != 1:
                raise ValueError("Expected one shared-input class-routed dense layer.")
        self.rewirers = [
            _SingleLayerLossProbedRewirer(model, config, index, layer)
            for index, layer in targets
        ]
        self.config = config
        self.events = 0

    @property
    def num_layers(self) -> int:
        return len(self.rewirers)

    def maybe_rewire(
        self,
        model: torch.nn.Sequential,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
        *,
        probe_inputs: torch.Tensor,
        probe_targets: torch.Tensor,
        layer_probe_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict | None:
        if step < self.config.rewire_warmup or step % self.config.rewire_interval:
            return None
        count = len(self.rewirers)
        budget, remainder = divmod(self.config.rewire_max_rows, count)
        events = []
        for position, rewirer in enumerate(self.rewirers):
            layer_probe_inputs, layer_probe_targets = (
                layer_probe_batches[position]
                if layer_probe_batches is not None
                else (probe_inputs, probe_targets)
            )
            # Rotate the remainder so every layer is visited even with a
            # global row budget smaller than the number of layers.
            rewirer.max_rows = budget + (
                (position - self.events) % count < remainder
            )
            event = rewirer.maybe_rewire(
                model, inputs, targets, step, optimizer,
                probe_inputs=layer_probe_inputs,
                probe_targets=layer_probe_targets,
            )
            if event is not None:
                events.append(event)
        self.events += 1
        if count == 1:
            return events[0]
        return {
            "step": step,
            "mean_entropy": sum(event["mean_entropy"] for event in events) / count,
            "converged_rows": sum(event["converged_rows"] for event in events),
            "stalled_rows": sum(event["stalled_rows"] for event in events),
            "eligible_rows": sum(event["eligible_rows"] for event in events),
            "attempted_rows": sum(event["attempted_rows"] for event in events),
            "accepted_rows": sum(event["accepted_rows"] for event in events),
            "search_ce_before": None,
            "search_ce_after": None,
            "probe_ce_before": None,
            "probe_ce_after": None,
            "elapsed_seconds": sum(event["elapsed_seconds"] for event in events),
        }

    def summary(self) -> dict:
        summaries = [rewirer.summary() for rewirer in self.rewirers]
        if len(summaries) == 1:
            return summaries[0]
        return {
            "events": self.events,
            "attempted_rows": sum(item["attempted_rows"] for item in summaries),
            "accepted_rows": sum(item["accepted_rows"] for item in summaries),
            "elapsed_seconds": sum(item["elapsed_seconds"] for item in summaries),
            "layers": [
                {"model_index": rewirer.layer_index, **summary}
                for rewirer, summary in zip(self.rewirers, summaries)
            ],
        }


REWIRING_FIELDS = (
    "step", "mean_entropy", "converged_rows", "stalled_rows",
    "eligible_rows", "attempted_rows", "accepted_rows",
    "search_ce_before", "search_ce_after", "probe_ce_before",
    "probe_ce_after", "elapsed_seconds",
)
