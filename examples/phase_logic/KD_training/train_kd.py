"""Baseline and logit KD training. Never invoked automatically on import."""

from dataclasses import asdict, fields
import csv
from contextlib import ExitStack
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import Config, ROOT, parse_args
from .run_naming import make_run_name
from .student import (
    LUT6Student,
    TinyConvLUT6Student,
    TinyConvWide10LUT6Student,
    TinyConv2DenseLUT6Student,
    TinyConv2DenseWide10LUT6Student,
    TinyConvQ2LBP2Dense2_6000LUT6Student,
    TinyConvQ2OneConvDense2LUT6Student,
    TinyConvQ2Wide10Dense2LUT6Student,
    TinyConvQ2Wide10Dense2FourConvLUT6Student,
    TinyConvQ2Compact20Dense2LUT6Student,
    TinyConvLBP2Wide10Dense2LUT6Student,
)
from .staged_freezing import DiscreteStabilityStagedFreezer, STAGED_FREEZE_FIELDS
from .kd_infusion_training import KDProgressiveAggressiveInfusion, INFUSION_LOG_FIELDS
from .kd_infusion_training.checkpoint import load_training_checkpoint, save_training_checkpoint
from .teacher import FrozenTeacher, read_teacher_config
from ..config import ExperimentConfig


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def cifar_transform(saved, augment):
    """Mirror data._build_cifar10 transforms, without its download=True loader."""
    known = {field.name for field in fields(ExperimentConfig)}
    source = ExperimentConfig(**{key: value for key, value in saved.items() if key in known})
    steps = [transforms.ToTensor()]
    if augment and source.cifar_augmentation:
        if source.uses_compact_logic:
            steps = [transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(32, padding=2, padding_mode="edge"), transforms.ToTensor()]
        else:
            steps += [transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                      transforms.RandomHorizontalFlip()]
        if source.cifar_random_erasing:
            steps.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.12),
                                                  ratio=(0.5, 2.0), value=0.0))
    return transforms.Compose(steps)


def data_loaders(config, saved):
    # The existing pipeline has no validation split and evaluates CIFAR test
    # data during training. Use a fixed holdout from training data instead.
    augmented = datasets.CIFAR10(config.data_dir, train=True, download=False,
                                transform=cifar_transform(saved, config.augment))
    plain = datasets.CIFAR10(config.data_dir, train=True, download=False,
                            transform=cifar_transform(saved, False))
    order = torch.randperm(len(plain), generator=torch.Generator().manual_seed(config.split_seed))
    validation_ids, train_ids = order[:config.validation_size], order[config.validation_size:]
    options = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                   pin_memory=torch.device(config.device).type == "cuda", worker_init_fn=seed_worker)
    train = DataLoader(Subset(augmented, train_ids.tolist()), shuffle=True,
                       generator=torch.Generator().manual_seed(config.seed), **options)
    validation = DataLoader(Subset(plain, validation_ids.tolist()), shuffle=False,
                            generator=torch.Generator().manual_seed(config.seed + 1), **options)
    return train, validation, {"train_indices": train_ids, "validation_indices": validation_ids}


@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    loss = correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = student(images)
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += int((logits.argmax(-1) == labels).sum())
        count += labels.numel()
    return {"loss": loss / count, "accuracy": correct / count}


def load_best_for_continuation(path, config):
    """Validate an existing best checkpoint before allocating a new run."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not all(
        key in checkpoint for key in ("epoch", "student_state", "optimizer_state", "config", "validation")
    ):
        raise ValueError("--resume-from-best requires a KD best_student.pt training checkpoint.")
    previous = checkpoint["config"]
    current = json.loads(json.dumps(asdict(config), default=str))
    mutable = {"epochs", "run_name", "output_dir", "device", "num_workers",
               "teacher_backend", "evaluate_test", "resume_from_best"}
    for name, value in current.items():
        if name not in mutable and previous.get(name, getattr(Config(), name)) != value:
            raise ValueError(f"Resume setting differs from the checkpoint: {name}.")
    if config.epochs <= checkpoint["epoch"]:
        raise ValueError("--epochs must exceed the checkpoint's saved epoch.")
    return checkpoint


def main():
    config = parse_args()
    config.run_name = make_run_name(config)
    resume_best = (load_best_for_continuation(config.resume_from_best, config)
                   if config.resume_from_best is not None else None)
    # Reject existing/unsafe destinations before even deserializing the teacher.
    run = config.output_dir.expanduser().resolve() / config.run_name
    if not run.resolve().is_relative_to(ROOT):
        raise ValueError("Run directory must stay inside KD_training.")
    if run.exists() or run.is_symlink():
        raise FileExistsError(f"Run directory already exists: {run}. Use a new --run-name.")
    # CPU metadata/state validation precedes dataset and model construction.
    saved_config = read_teacher_config(config.teacher)
    known = {field.name for field in fields(ExperimentConfig)}
    source = ExperimentConfig(**{key: value for key, value in saved_config.items() if key in known})
    if source.dataset != "cifar10" or source.parametrization != "compact" or not (
        source.uses_compact_logic or source.uses_classrouted_compact_head
    ):
        raise ValueError("This experiment requires a supported CompactLogic CIFAR-10 teacher.")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    # Reserve a fresh writable destination atomically before expensive setup.
    run.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train, validation, split = data_loaders(config, saved_config)
    options = dict(mapping_candidates=config.mapping_candidates,
                   mapping_temperature=config.mapping_temperature,
                   student_tau=config.student_tau, node_chunk=config.node_chunk)
    if config.student_model in {
        "tinyconv_q2_lbp2", "tinyconv_q2_lbp2_wide10", "tinyconv_q2_lbp2_dense2",
        "tinyconv_q2_lbp2_dense2_wide10",
        "tinyconv_q2_lbp2_dense2_6000",
        "tinyconv_q2_1conv_dense2",
        "tinyconv_q2_wide10_dense2", "tinyconv_q2_wide10_dense2_4conv",
        "tinyconv_q2_compact20_dense2",
        "tinyconv_lbp2_wide10_dense2",
    }:
        thresholds = None
        if config.student_model != "tinyconv_lbp2_wide10_dense2":
            from ..data import compute_cifar10_quantile_thresholds

            # Fit only unaugmented student-training pixels, never validation/test.
            raw_training = SimpleNamespace(train=True,
                data=train.dataset.dataset.data[split["train_indices"].numpy()])
            thresholds = compute_cifar10_quantile_thresholds(raw_training, num_bits=2)
            del raw_training
        student_class = {
            "tinyconv_q2_lbp2": TinyConvLUT6Student,
            "tinyconv_q2_lbp2_wide10": TinyConvWide10LUT6Student,
            "tinyconv_q2_lbp2_dense2": TinyConv2DenseLUT6Student,
            "tinyconv_q2_lbp2_dense2_wide10": TinyConv2DenseWide10LUT6Student,
            "tinyconv_q2_lbp2_dense2_6000": TinyConvQ2LBP2Dense2_6000LUT6Student,
            "tinyconv_q2_1conv_dense2": TinyConvQ2OneConvDense2LUT6Student,
            "tinyconv_q2_wide10_dense2": TinyConvQ2Wide10Dense2LUT6Student,
            "tinyconv_q2_wide10_dense2_4conv": TinyConvQ2Wide10Dense2FourConvLUT6Student,
            "tinyconv_q2_compact20_dense2": TinyConvQ2Compact20Dense2LUT6Student,
            "tinyconv_lbp2_wide10_dense2": TinyConvLBP2Wide10Dense2LUT6Student,
        }[config.student_model]
        student = student_class(thresholds=thresholds, **options).to(device)
    else:
        student = LUT6Student(**options).to(device)
    # Only KD instantiates the teacher. Both modes start with identical students.
    teacher = (FrozenTeacher(config.teacher, device=device, backend=config.teacher_backend)
               if config.mode == "kd" else None)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    infusion = (KDProgressiveAggressiveInfusion(
        student, optimizer, run / "kd_infusion_safe.pt",
        warmup_epochs=config.kd_infusion_warmup_epochs,
        interval_epochs=config.kd_infusion_interval_epochs,
        fraction=config.kd_infusion_fraction,
        minimum_transition_steps=config.kd_infusion_cooldown_steps,
    ) if config.kd_progressive_aggressive_infusion else None)
    staged_freezer = (
        DiscreteStabilityStagedFreezer(
            student, optimizer, run / "staged_freezing_safe.pt",
        )
        if config.discrete_stability_staged_freezing is not None else None
    )

    # Keep checkpoints weights_only-loadable: Path objects are not safe globals.
    checkpoint_config = json.loads(json.dumps(asdict(config), default=str))
    metadata = dict(checkpoint_config)
    metadata.update(student_parameter_count=sum(p.numel() for p in student.parameters()),
                    student_architecture=("LBP2 -> LUT6Conv35/70 (3x3,s2) -> 8800 LUT6 -> 8800 LUT6 -> GroupSum(10, 880/class)"
                                          if config.student_model == "tinyconv_lbp2_wide10_dense2" else
                                          "Q2 -> LUT6Conv28/56 (3x3,s2) -> 7040 LUT6 -> 7040 LUT6 -> GroupSum(10, 704/class)"
                                          if config.student_model == "tinyconv_q2_compact20_dense2" else
                                          "Q2 -> LUT6Conv35/70/105/140 (3x3,s2) -> 8800 LUT6 -> 8800 LUT6 -> GroupSum(10, 880/class)"
                                          if config.student_model == "tinyconv_q2_wide10_dense2_4conv" else
                                          "Q2 -> LUT6Conv35/70 (3x3,s2) -> 8800 LUT6 -> 8800 LUT6 -> GroupSum(10, 880/class)"
                                          if config.student_model == "tinyconv_q2_wide10_dense2" else
                                          "Q2+LBP2 -> LUT6Conv32/64 (3x3,s2) -> 8800 LUT6 -> 8800 LUT6 -> GroupSum(10, 880/class)"
                                          if config.student_model == "tinyconv_q2_lbp2_dense2_wide10" else
                                          "Q2+LBP2 -> LUT6Conv32/64 (3x3,s2) -> 6000 LUT6 -> 6000 LUT6 -> GroupSum(10, 600/class)"
                                          if config.student_model == "tinyconv_q2_lbp2_dense2_6000" else
                                          "Q2 -> LUT6Conv32 (3x3,s2) -> 8000 LUT6 -> 8000 LUT6 -> GroupSum(10, 800/class)"
                                          if config.student_model == "tinyconv_q2_1conv_dense2" else
                                          "Q2+LBP2 -> LUT6Conv32/64 (3x3,s2) -> 8000 LUT6 -> 8000 LUT6 -> GroupSum(10, 800/class)"
                                          if config.student_model == "tinyconv_q2_lbp2_dense2" else
                                          "Q2+LBP2 -> LUT6Conv35/70 (3x3,s2) -> 8800 LUT6 -> GroupSum(10, 880/class)"
                                          if config.student_model == "tinyconv_q2_lbp2_wide10" else
                                          "Q2+LBP2 -> LUT6Conv32/64 (3x3,s2) -> 8000 LUT6 -> GroupSum(10, 800/class)"
                                          if config.student_model == "tinyconv_q2_lbp2" else
                                          "thermometer10 -> 8000 LUT6 -> GroupSum(10, 800/class)"),
                    teacher_mode="discrete eval" if teacher else "not loaded",
                    preprocessing=repr(train.dataset.dataset.transform),
                    teacher_training_config=saved_config,
                    teacher_group_sum_tau=teacher.report["group_sum_tau"] if teacher else None,
                    mapping="bounded disjoint candidates, hard forward / softmax surrogate",
                    lut_gradient="multilinear vertex derivative + output STE; not DWN EFD",
                    staged_freezing=(staged_freezer.summary() if staged_freezer else None))
    if infusion is not None:
        metadata["kd_progressive_aggressive_infusion"] = infusion.summary()
    (run / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    torch.save(split, run / "split.pt")
    print(f"Student parameters={metadata['student_parameter_count']:,}; results={run}", flush=True)
    if teacher:
        print(f"Frozen teacher tau={teacher.report['group_sum_tau']}; student tau={config.student_tau}", flush=True)
    best_accuracy, best_loss = -1.0, float("inf")
    start_epoch, global_step = 1, 0
    if resume_best is not None:
        student.load_state_dict(resume_best["student_state"], strict=True)
        optimizer.load_state_dict(resume_best["optimizer_state"])
        start_epoch = int(resume_best["epoch"]) + 1
        global_step = (start_epoch - 1) * len(train)
        best_accuracy = float(resume_best["validation"]["accuracy"])
        best_loss = float(resume_best["validation"]["loss"])
        if staged_freezer is not None:
            frozen = resume_best.get("staged_freezing", {}).get("frozen_mappings", [])
            names = [name for name, _ in staged_freezer.mapping_phases]
            if frozen != names[:len(frozen)]:
                raise ValueError("Best checkpoint has incompatible staged freezing phases.")
            staged_freezer.phase_index = len(frozen)
            staged_freezer._apply_freeze_state()
            # An unfinished monitoring window cannot be reconstructed from the
            # best checkpoint; resume observing the next unfrozen mapping.
            if not staged_freezer.complete:
                staged_freezer.previous_signature = staged_freezer._signature()
        if infusion is not None:
            saved_infusion = resume_best.get("kd_infusion", {})
            infusion.level = int(saved_infusion.get("level", 1))
            infusion.total_proposed_rows = int(saved_infusion.get("total_proposed_rows", 0))
            infusion.best_accuracy = best_accuracy
        torch.save(resume_best, run / "best_student.pt")
        metadata.update(resumed_from=str(config.resume_from_best), resume_epoch=start_epoch,
                        resume_kind="best_checkpoint_weights_and_optimizer",
                        resume_note="RNG, data-loader order, and pending controller monitoring restart.")
        (run / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        print(f"Continued from best checkpoint at epoch={start_epoch}", flush=True)
    if infusion is not None and config.kd_infusion_resume is not None:
        resumed = load_training_checkpoint(
            config.kd_infusion_resume, controller=infusion, config=checkpoint_config,
            best_path=run / "best_student.pt", loaders=(train, validation),
        )
        start_epoch, global_step = resumed["epoch"] + 1, resumed["step"]
        best_metrics = resumed["best_checkpoint"]["validation"]
        best_accuracy, best_loss = best_metrics["accuracy"], best_metrics["loss"]
        metadata["kd_progressive_aggressive_infusion"] = infusion.summary()
        metadata["resumed_from"] = str(config.kd_infusion_resume)
        metadata["resume_epoch"] = start_epoch
        (run / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        print(f"Resumed KD infusion at epoch={start_epoch} level={infusion.level}", flush=True)
    columns = ["epoch", "train_loss", "train_ce", "train_kl", "train_accuracy",
               "validation_loss", "validation_accuracy"]
    if infusion is not None:
        columns.append("infusion_level")
    with ExitStack() as stack:
        stream = stack.enter_context((run / "metrics.csv").open("x", newline=""))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        freeze_writer = None
        freeze_stream = None
        if staged_freezer is not None:
            freeze_stream = stack.enter_context(
                (run / "staged_freezing.csv").open("x", newline="")
            )
            freeze_writer = csv.DictWriter(freeze_stream, fieldnames=STAGED_FREEZE_FIELDS)
            freeze_writer.writeheader()
        infusion_writer = None
        if infusion is not None:
            infusion_stream = stack.enter_context((run / "kd_infusion.csv").open("x", newline=""))
            infusion_writer = csv.DictWriter(infusion_stream, fieldnames=INFUSION_LOG_FIELDS)
            infusion_writer.writeheader()
        for epoch in range(start_epoch, config.epochs + 1):
            student.train()
            total = ce_total = kl_total = correct = seen = 0
            for images, labels in train:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = student(images)
                ce = F.cross_entropy(logits, labels)
                kl = logits.new_zeros(())
                if teacher:
                    target = teacher(images)
                    if target.shape != logits.shape:
                        raise ValueError("Teacher/student logits must both have shape [batch,10].")
                    temperature = config.temperature
                    kl = F.kl_div(F.log_softmax(logits / temperature, dim=-1),
                                  F.softmax(target / temperature, dim=-1), reduction="batchmean")
                    loss = (1 - config.alpha) * ce + config.alpha * temperature**2 * kl
                else:
                    loss = ce
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; aborting without saving invalid weights.")
                loss.backward()
                optimizer.step()
                student.clamp_tables()
                global_step += 1
                size = labels.numel()
                total += loss.item() * size
                ce_total += ce.item() * size
                kl_total += kl.item() * size
                correct += int((logits.detach().argmax(-1) == labels).sum())
                seen += size
            metrics = evaluate(student, validation, device)
            row = dict(zip(columns, [epoch, total / seen, ce_total / seen, kl_total / seen,
                                     correct / seen, metrics["loss"], metrics["accuracy"]]))
            if infusion is not None:
                row["infusion_level"] = infusion.level
            writer.writerow(row)
            stream.flush()
            print(f"epoch={epoch} train_loss={row['train_loss']:.4f} "
                  f"train_acc={row['train_accuracy']:.4f} val_loss={metrics['loss']:.4f} "
                  f"val_acc_discrete={metrics['accuracy']:.4f}"
                  + (f" infusion_level={infusion.level}" if infusion is not None else ""), flush=True)
            allow_best_checkpoint = True
            infusion_event = None
            if infusion is not None:
                infusion_event = infusion.observe(
                    epoch=epoch, step=global_step, accuracy=metrics["accuracy"], loss=metrics["loss"],
                )
                infusion_writer.writerow(infusion_event)
                infusion_stream.flush()
                allow_best_checkpoint = infusion_event["action"] != "rollback"
                if infusion_event["action"] in {"advance", "rollback"}:
                    print(f"  kd_infusion {infusion_event['action']} "
                          f"level={infusion_event['from_level']}->{infusion_event['to_level']} "
                          f"epoch={epoch}", flush=True)
            if (staged_freezer is not None
                    and epoch > config.discrete_stability_staged_freezing):
                event = staged_freezer.observe(
                    epoch=epoch, accuracy=metrics["accuracy"], loss=metrics["loss"],
                )
                freeze_writer.writerow({field: event[field] for field in STAGED_FREEZE_FIELDS})
                freeze_stream.flush()
                allow_best_checkpoint = event["allow_best_checkpoint"]
                if event["action"] in {"freeze_start", "accept", "rollback"}:
                    print(
                        f"  staged_freeze epoch={epoch} target={event['target']} "
                        f"action={event['action']} reason={event['reason']}",
                        flush=True,
                    )
            if (allow_best_checkpoint
                    and (metrics["accuracy"], -metrics["loss"]) > (best_accuracy, -best_loss)):
                best_accuracy, best_loss = metrics["accuracy"], metrics["loss"]
                torch.save({"epoch": epoch, "student_state": student.state_dict(),
                            "optimizer_state": optimizer.state_dict(), "config": checkpoint_config,
                            "validation": metrics,
                            **({"kd_infusion": infusion.summary()} if infusion is not None else {}),
                            **({"staged_freezing": staged_freezer.summary()}
                               if staged_freezer is not None else {})}, run / "best_student.pt")
            if infusion is not None:
                if infusion_event["action"] != "rollback" and epoch < config.epochs:
                    proposals = infusion.infuse(epoch=epoch, step=global_step)
                    infusion_writer.writerows(proposals)
                    infusion_stream.flush()
                    if proposals:
                        print(f"  kd_infusion level={infusion.level} "
                              f"proposed_rows={sum(event['rows'] for event in proposals)}", flush=True)
                save_training_checkpoint(
                    run / "kd_infusion_latest.pt", controller=infusion, config=checkpoint_config,
                    epoch=epoch, step=global_step, best_path=run / "best_student.pt",
                    loaders=(train, validation),
                )
    infusion_end = infusion.summary() if infusion is not None else None
    # Reconstruct discrete choices from the validation-selected student only.
    best = torch.load(run / "best_student.pt", map_location=device, weights_only=True)
    student.load_state_dict(best["student_state"], strict=True)
    student.eval()
    torch.save(student.discrete_state(), run / "discrete_student.pt")
    summary = {"best_epoch": best["epoch"], "validation_discrete": evaluate(student, validation, device),
               "test_evaluated": config.evaluate_test}
    if staged_freezer is not None:
        summary["staged_freezing"] = {
            "best_checkpoint": best.get("staged_freezing"),
            "training_end": staged_freezer.summary(),
        }
    if infusion is not None:
        summary["kd_progressive_aggressive_infusion"] = {
            "best_checkpoint": best.get("kd_infusion"), "training_end": infusion_end,
        }
    if config.evaluate_test:
        test_set = datasets.CIFAR10(config.data_dir, train=False, download=False,
                                   transform=cifar_transform(saved_config, False))
        test = DataLoader(test_set, batch_size=config.batch_size, num_workers=config.num_workers)
        summary["test_discrete"] = evaluate(student, test, device)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
