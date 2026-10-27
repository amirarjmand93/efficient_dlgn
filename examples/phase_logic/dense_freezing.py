"""Opt-in, layer-ordered dense group freezing after convolution freezing."""

import math

import torch
from torch.nn import functional as F

from .compact import CandidateLogicLayer, _boolean_operation


DENSE_FREEZE_FIELDS = (
    "step", "layer", "group_size", "loss_soft", "loss_hard", "loss_increase",
    "max_batch_loss_increase", "loss_stderr", "loss_upper_bound", "probe_batches",
    "probe_examples", "passed", "consecutive_passes", "committed",
    "frozen_rows", "total_rows", "layer_frozen",
)


class GroupwiseCandidateDense(CandidateLogicLayer):
    """Wrap an existing engine without reinitializing its weights or connections.

    Frozen outputs use their saved Boolean gate/pair, not a softmax mixture.
    Optimizer hooks additionally restore their logits and zero Adam moments,
    protecting against both momentum and weight decay. Hard gate evaluation
    remains differentiable with respect to inputs through its Boolean extension.
    """

    def __init__(self, original):
        torch.nn.Module.__init__(self)
        if original.is_convolution or original.is_frozen:
            raise ValueError("Dense group freezing requires an unfrozen dense candidate layer.")
        for name in ("backend", "num_gates", "lut_rank", "out_dim", "in_shape", "stride"):
            setattr(self, name, getattr(original, name))
        self.is_convolution = False
        self.impl = original.impl
        self.register_buffer("frozen", original.frozen.detach().clone())
        self.register_buffer("frozen_rows", torch.zeros(self.out_dim, dtype=torch.bool,
                                                       device=self.weights.device))
        self.register_buffer("frozen_logits", torch.zeros_like(self.weights))
        self.register_buffer("frozen_codes", torch.zeros(self.out_dim, dtype=torch.uint8,
                                                        device=self.weights.device))
        self.register_buffer("frozen_pairs", torch.zeros(2, self.out_dim, dtype=torch.long,
                                                        device=self.weights.device))
        self.register_buffer("frozen_row_indices", torch.empty(0, dtype=torch.long,
                                                              device=self.weights.device), persistent=False)
        self._is_frozen = False
        self.num_frozen_rows = 0
        self.register_load_state_dict_post_hook(self._restore_group_state)
        self.train(original.training)

    def _restore_group_state(self, module, incompatible_keys):
        self.frozen_row_indices = self.frozen_rows.nonzero(as_tuple=True)[0]
        self.num_frozen_rows = self.frozen_row_indices.numel()
        self._is_frozen = bool(self.frozen)
        if self._is_frozen != (self.num_frozen_rows == self.out_dim):
            raise ValueError("Dense frozen flag and frozen-row mask disagree.")
        self.weights.requires_grad_(not self._is_frozen)
        self.impl.train(self.training and not self._is_frozen)

    @staticmethod
    def hard_rows(x, pairs, codes):
        return _boolean_operation(x[:, pairs[0]], x[:, pairs[1]], codes)

    def forward(self, x):
        if not self.training or not self.num_frozen_rows:
            return super().forward(x)
        rows = self.frozen_row_indices
        hard = self.hard_rows(x, self.frozen_pairs[:, rows], self.frozen_codes[rows])
        if self.is_frozen:
            return hard
        soft = super().forward(x)
        return soft.index_copy(1, rows, hard)

    @torch.no_grad()
    def proposal(self, rows):
        rows = rows.detach().clone()
        winners = self.weights[rows].argmax(-1)
        return {
            "rows": rows,
            "winners": winners,
            "logits": self.weights[rows].detach().clone(),
            "codes": self.impl.gate_sequence[rows, winners].clone(),
            "pairs": self.impl.indices[:, rows, winners].clone(),
        }

    @torch.no_grad()
    def freeze_rows(self, proposal, optimizer=None):
        rows, winners = proposal["rows"], proposal["winners"]
        if rows.numel() == 0 or bool(self.frozen_rows[rows].any()):
            raise ValueError("A dense freeze proposal must contain unfrozen rows.")
        if rows.unique().numel() != rows.numel():
            raise ValueError("Dense freeze rows must be distinct.")
        if not torch.equal(proposal["logits"].argmax(-1), winners):
            raise ValueError("Dense proposal logits do not match their pinned winners.")
        changed_pairs = not torch.equal(self.impl.indices[:, rows, winners], proposal["pairs"])
        self.impl.indices[:, rows, winners] = proposal["pairs"]
        self.impl.gate_sequence[rows, winners] = proposal["codes"]
        self.weights[rows] = proposal["logits"]
        self.frozen_logits[rows] = proposal["logits"]
        self.frozen_codes[rows] = proposal["codes"]
        self.frozen_pairs[:, rows] = proposal["pairs"]
        self.frozen_rows[rows] = True
        self.frozen_row_indices = self.frozen_rows.nonzero(as_tuple=True)[0]
        self.num_frozen_rows = self.frozen_row_indices.numel()
        if changed_pairs and self.backend == "cuda":
            self.impl.refresh_backward_index(rows)
        if self.num_frozen_rows == self.out_dim:
            super().freeze()
            self.weights.grad = None
        if optimizer is not None:
            self.protect_optimizer_rows(optimizer)

    @torch.no_grad()
    def freeze(self):
        rows = (~self.frozen_rows).nonzero(as_tuple=True)[0]
        if rows.numel():
            self.freeze_rows(self.proposal(rows))

    @torch.no_grad()
    def protect_optimizer_rows(self, optimizer):
        if not self.num_frozen_rows:
            return
        rows = self.frozen_row_indices
        self.weights[rows] = self.frozen_logits[rows]
        if self.weights.grad is not None:
            self.weights.grad[rows] = 0
        state = optimizer.state.get(self.weights, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
            if key in state:
                state[key][rows] = 0

    @torch.no_grad()
    def resample(self, rows, optimizer):
        rows = rows[~self.frozen_rows[rows]]
        if not rows.numel():
            return 0, 0
        return super().resample(rows, optimizer)


def enable_dense_group_freezing(model):
    """Opt-in conversion preserving module paths, parameters, and baseline RNG."""
    for name, child in list(model.named_children()):
        if isinstance(child, GroupwiseCandidateDense):
            continue
        if isinstance(child, CandidateLogicLayer) and not child.is_convolution:
            model.add_module(name, GroupwiseCandidateDense(child))
        else:
            enable_dense_group_freezing(child)


class DenseFreezeController:
    """Probe fixed groups sequentially; never advance past an unfrozen layer."""

    def __init__(self, model, config, optimizer):
        self.config = config
        self.layers = [(name, layer) for name, layer in model.named_modules()
                       if isinstance(layer, GroupwiseCandidateDense)]
        self.convs = [layer for layer in model.modules()
                      if isinstance(layer, CandidateLogicLayer) and layer.is_convolution]
        if not self.layers:
            raise ValueError("Dense group-freezing layers must be installed before creating Adam.")
        self.conv_finished_step = None
        self.last_commit_step = None
        self.proposal = None
        self.confirmations = 0
        self.pending_probe = False
        self.last_probe = None
        self._optimizer_handles = [
            optimizer.register_step_pre_hook(self._protect_before_step),
            optimizer.register_step_post_hook(self._protect_after_step),
        ]

    def _protect_before_step(self, optimizer, args, kwargs):
        for _, layer in self.layers:
            layer.protect_optimizer_rows(optimizer)

    def _protect_after_step(self, optimizer, args, kwargs):
        for _, layer in self.layers:
            layer.protect_optimizer_rows(optimizer)

    @property
    def fully_frozen(self):
        return all(layer.is_frozen for _, layer in self.layers) and all(layer.is_frozen for layer in self.convs)

    @torch.no_grad()
    def step(self, step):
        self.pending_probe = False
        if step <= self.config.compact_exploration_steps or not all(layer.is_frozen for layer in self.convs):
            return
        if self.conv_finished_step is None:
            self.conv_finished_step = self.last_commit_step = step
        if self.fully_frozen:
            return
        if (step - self.last_commit_step < self.config.dense_freeze_min_interval
                or step % self.config.dense_freeze_probe_interval):
            return
        if self.proposal is None:
            name, layer = next((name, layer) for name, layer in self.layers if not layer.is_frozen)
            remaining = (~layer.frozen_rows).nonzero(as_tuple=True)[0]
            size = min(remaining.numel(), math.ceil(layer.out_dim * self.config.dense_freeze_fraction))
            priority = torch.argsort(layer.weights_entropy()[remaining], stable=True)
            self.proposal = {"layer": name, **layer.proposal(remaining[priority[:size]])}
            self.confirmations = 0
        self.pending_probe = True

    @staticmethod
    def _probe_batch(model, layer, proposal, inputs, targets, restore_buffers):
        """Paired CE intervention; same RNG/buffer conditions for both passes."""
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None
        captured = {}

        def capture(module, args, output):
            captured["hard"] = layer.hard_rows(args[0], proposal["pairs"], proposal["codes"])

        handle = layer.register_forward_hook(capture)
        try:
            soft = F.cross_entropy(model(inputs), targets, reduction="none")
        finally:
            handle.remove()
        restore_buffers()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, inputs.device)
        handle = layer.register_forward_hook(
            lambda module, args, output: output.index_copy(1, proposal["rows"], captured["hard"]))
        try:
            hard = F.cross_entropy(model(inputs), targets, reduction="none")
        finally:
            handle.remove()
        return {
            "soft": float(soft.mean()), "hard": float(hard.mean()),
            "changes": (hard - soft).cpu().double(), "count": targets.numel(),
        }

    @torch.no_grad()
    def probe_and_freeze(self, model, batches, step, optimizer):
        if not self.pending_probe:
            return None
        if len(batches) != self.config.dense_freeze_probe_batches:
            raise ValueError("Dense freezing requires dense_freeze_probe_batches batches.")
        device = batches[0][0].device
        if any(y.numel() == 0 or x.shape[0] != y.numel() or x.device != device or y.device != device
               for x, y in batches):
            raise ValueError("Dense probe batches must be nonempty, aligned, and on the same device.")
        self.pending_probe = False
        proposal = self.proposal
        name = proposal["layer"]
        layer = dict(self.layers)[name]
        modes = [(module, module.training) for module in model.modules()]
        buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]

        def restore_buffers():
            for buffer, saved in buffers:
                buffer.copy_(saved)

        try:
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                model.train()
                results = []
                for inputs, targets in batches:
                    restore_buffers()
                    results.append(self._probe_batch(model, layer, proposal, inputs, targets, restore_buffers))
        finally:
            restore_buffers()
            for module, training in modes:
                module.training = training
        changes = torch.cat([result["changes"] for result in results])
        count = changes.numel()
        increase = float(changes.mean())
        stderr = float(changes.std(unbiased=True) / math.sqrt(count)) if count > 1 else math.inf
        upper = increase + self.config.dense_freeze_loss_sigma * stderr
        max_increase = max(float(result["changes"].mean()) for result in results)
        finite = (all(math.isfinite(result[key]) for result in results for key in ("soft", "hard"))
                  and bool(torch.isfinite(changes).all()) and math.isfinite(upper))
        passed = (finite and max_increase <= self.config.dense_freeze_max_loss_increase
                  and upper <= self.config.dense_freeze_max_loss_increase)
        self.confirmations = self.confirmations + 1 if passed else 0
        committed = self.confirmations >= self.config.dense_freeze_probe_patience
        if committed:
            layer.freeze_rows(proposal, optimizer)
            self.last_commit_step = step
        self.last_probe = {
            "step": step, "layer": name, "group_size": proposal["rows"].numel(),
            "loss_soft": sum(r["soft"] * r["count"] for r in results) / count,
            "loss_hard": sum(r["hard"] * r["count"] for r in results) / count,
            "loss_increase": increase, "max_batch_loss_increase": max_increase,
            "loss_stderr": stderr, "loss_upper_bound": upper,
            "probe_batches": len(batches), "probe_examples": count,
            "passed": passed, "consecutive_passes": self.confirmations, "committed": committed,
            "frozen_rows": layer.num_frozen_rows, "total_rows": layer.out_dim,
            "layer_frozen": layer.is_frozen,
        }
        for key in ("loss_soft", "loss_hard", "loss_increase", "max_batch_loss_increase",
                    "loss_stderr", "loss_upper_bound"):
            if not math.isfinite(self.last_probe[key]):
                self.last_probe[key] = None
        if committed or not passed:
            self.proposal = None
            self.confirmations = 0
        return dict(self.last_probe)

    def summary(self):
        return {
            "mode": "dense_groups_v1", "group_fraction": self.config.dense_freeze_fraction,
            "conv_finished_step": self.conv_finished_step,
            "next_layer": next((name for name, layer in self.layers if not layer.is_frozen), None),
            "layers": {name: {"frozen_rows": layer.num_frozen_rows, "total_rows": layer.out_dim}
                       for name, layer in self.layers},
            "last_probe": self.last_probe,
        }

    def state_dict(self):
        return {
            "conv_finished_step": self.conv_finished_step, "last_commit_step": self.last_commit_step,
            "confirmations": self.confirmations, "last_probe": self.last_probe,
            "proposal": ({key: value.clone() if isinstance(value, torch.Tensor) else value
                          for key, value in self.proposal.items()} if self.proposal is not None else None),
        }

    def load_state_dict(self, state):
        self.conv_finished_step = state["conv_finished_step"]
        self.last_commit_step = state["last_commit_step"]
        self.confirmations = state["confirmations"]
        self.last_probe = state["last_probe"]
        proposal = state["proposal"]
        if proposal is not None:
            device = dict(self.layers)[proposal["layer"]].weights.device
            proposal = {key: value.clone().to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in proposal.items()}
        self.proposal = proposal
        self.pending_probe = False
