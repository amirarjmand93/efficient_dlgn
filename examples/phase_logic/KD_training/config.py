"""Configuration for the isolated 8,000-LUT6 distillation experiment."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

from .student import STUDENT_MODELS


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DEFAULT_TEACHER = REPO / (
    "examples/phase_logic/results/bestof best/"
    "compact_shared_head2_v2_concat_w92400x92400_cuda_c16_explore700000_resample100_freeze200/"
    "efficient_layer_aware_resampling_v2_conv25_dense1_u70_i18_n8_c4_dense2_u80_i12_n6_c2/"
    "hybrid_freeze_v3_probedriven_n4_sigma2_gap5000_probe1000_confirm3_drift1_flip0.001_taylor0.2_loss0.08/"
    "compact/build_model_cifar10_small_compact_head/pure_ce/seed_4242449/best_model.pt"
)


@dataclass
class Config:
    mode: str = "baseline"
    student_model: str = "lut6"
    teacher: Path = DEFAULT_TEACHER
    teacher_backend: str = "auto"
    data_dir: Path = REPO / "data"
    output_dir: Path = ROOT / "runs"
    run_name: str | None = None
    device: str = "cpu"
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.01
    seed: int = 42
    split_seed: int = 42
    validation_size: int = 5000
    num_workers: int = 4
    alpha: float = 0.5
    temperature: float = 2.0
    student_tau: float = 40.0
    mapping_candidates: int = 16
    mapping_temperature: float = 1.0
    node_chunk: int = 256
    augment: bool = True
    evaluate_test: bool = False
    discrete_stability_staged_freezing: int | None = None
    kd_progressive_aggressive_infusion: bool = False
    kd_infusion_warmup_epochs: int = 100
    kd_infusion_interval_epochs: int = 10
    kd_infusion_fraction: float = 0.01
    kd_infusion_cooldown_steps: int = 15_000
    kd_infusion_resume: Path | None = None
    resume_from_best: Path | None = None

    def validate(self):
        for key in ("epochs", "batch_size", "mapping_candidates", "node_chunk"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive.")
        if not 1 <= self.validation_size < 50000 or self.num_workers < 0:
            raise ValueError("Use 1..49999 validation images and nonnegative num_workers.")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        for key in ("temperature", "student_tau", "mapping_temperature", "learning_rate"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        if self.student_model not in STUDENT_MODELS:
            raise ValueError(f"Unknown student model: {self.student_model}")
        if self.kd_progressive_aggressive_infusion:
            if self.mode != "kd":
                raise ValueError("--kd-progressive-aggressive-infusion requires --mode kd.")
            if self.discrete_stability_staged_freezing is not None:
                raise ValueError("Test KD infusion separately from staged freezing; both restore safe checkpoints.")
        if self.kd_infusion_resume is not None and not self.kd_progressive_aggressive_infusion:
            raise ValueError("--kd-infusion-resume requires --kd-progressive-aggressive-infusion.")
        if self.kd_infusion_resume is not None and self.resume_from_best is not None:
            raise ValueError("Choose either --kd-infusion-resume or --resume-from-best.")
        if self.kd_infusion_warmup_epochs < 1 or self.kd_infusion_interval_epochs < 1:
            raise ValueError("Infusion warmup and interval epochs must be positive.")
        if (not math.isfinite(self.kd_infusion_fraction)
                or not 0 < self.kd_infusion_fraction <= 1 or self.kd_infusion_cooldown_steps < 0):
            raise ValueError("Infusion fraction must be in (0,1] and cooldown steps nonnegative.")
        if (self.discrete_stability_staged_freezing is not None
                and self.discrete_stability_staged_freezing < 1):
            raise ValueError(
                "--discrete-stability-staged-freezing epoch must be positive."
            )
        if (self.discrete_stability_staged_freezing
                and self.student_model not in {
                    "tinyconv_q2_lbp2_dense2", "tinyconv_q2_lbp2_dense2_wide10",
                    "tinyconv_q2_wide10_dense2",
                    "tinyconv_q2_lbp2_dense2_6000",
                    "tinyconv_q2_1conv_dense2",
                    "tinyconv_q2_wide10_dense2_4conv",
                    "tinyconv_q2_compact20_dense2",
                    "tinyconv_lbp2_wide10_dense2",
                }):
            raise ValueError(
                "--discrete-stability-staged-freezing requires "
                "a two-dense student model."
            )
        available = 30720 if self.student_model == "lut6" else 12 * 9
        if self.mapping_candidates * 6 > available:
            raise ValueError("Candidate pools must fit the student layer's available inputs.")
        if self.seed < 0 or self.split_seed < 0:
            raise ValueError("Seeds must be nonnegative.")
        output = self.output_dir.expanduser().resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError(f"All outputs must stay inside {ROOT}.")
        if self.run_name and (Path(self.run_name).name != self.run_name or self.run_name in {".", ".."}):
            raise ValueError("run_name must be a single directory name.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "kd"), default="baseline")
    parser.add_argument("--student-model", choices=STUDENT_MODELS, default="lut6")
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-backend", choices=("auto", "torch", "cuda"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-name", help="Override the automatic settings + UTC date/time directory name.")
    parser.add_argument("--device", default="cpu")
    for name, default in (("epochs", 100), ("batch-size", 128), ("seed", 42),
                          ("split-seed", 42), ("validation-size", 5000), ("num-workers", 4),
                          ("mapping-candidates", 16), ("node-chunk", 256)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", 0.01), ("alpha", 0.5), ("temperature", 2.0),
                          ("student-tau", 40.0), ("mapping-temperature", 1.0)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--no-augmentation", dest="augment", action="store_false")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Explicitly evaluate the best student on test data once, after training.")
    parser.add_argument(
        "--discrete-stability-staged-freezing", type=int, metavar="EPOCH",
        help=("Keep mappings exploring through EPOCH, then begin stability observation "
              "and staged freezing of Conv1 followed by Conv2, with validation "
              "monitoring and rollback."),
    )
    parser.add_argument(
        "--kd-progressive-aggressive-infusion", action="store_true",
        help="Enable validation-protected five-level truth-table proposals in dense LUT6 student layers.",
    )
    parser.add_argument("--kd-infusion-warmup-epochs", type=int, default=100)
    parser.add_argument("--kd-infusion-interval-epochs", type=int, default=10)
    parser.add_argument("--kd-infusion-fraction", type=float, default=0.01,
                        help="Maximum fraction of rows proposed per dense layer at each infusion event.")
    parser.add_argument("--kd-infusion-cooldown-steps", type=int, default=15_000)
    parser.add_argument("--kd-infusion-resume", type=Path,
                        help="Resume kd_infusion_latest.pt into a new run directory.")
    parser.add_argument("--resume-from-best", type=Path, metavar="BEST_STUDENT_PT",
                        help="Continue from a saved best_student.pt at its next epoch in a new run directory.")
    config = Config(**vars(parser.parse_args()))
    config.validate()
    return config
