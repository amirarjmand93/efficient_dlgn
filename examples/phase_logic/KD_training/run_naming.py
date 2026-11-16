"""Readable experiment names with UTC timestamps; no filesystem side effects."""

from datetime import datetime, timezone


def make_run_name(config):
    if config.run_name:
        return config.run_name
    parts = [
        "cifar10", config.student_model, config.mode, f"e{config.epochs}", f"b{config.batch_size}",
        f"lr{config.learning_rate:g}", f"c{config.mapping_candidates}",
        f"tau{config.student_tau:g}", f"seed{config.seed}",
    ]
    if config.mode == "kd":
        parts.extend((f"a{config.alpha:g}", f"T{config.temperature:g}"))
    if config.discrete_stability_staged_freezing is not None:
        parts.append(f"dsfreeze_after{config.discrete_stability_staged_freezing}")
    if getattr(config, "kd_progressive_aggressive_infusion", False):
        parts.append(f"kdinfusion_w{config.kd_infusion_warmup_epochs}"
                     f"_i{config.kd_infusion_interval_epochs}_f{config.kd_infusion_fraction:g}")
    parts.append(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    return "_".join(parts)
