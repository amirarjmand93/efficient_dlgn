# Codex Task: Implement Continuous Adaptive Infusion (CAI) — New Independent Training Method

## 1. Objective

Study my existing **Progressive Aggressive Infusion** training implementation carefully, then develop a new method called **Continuous Adaptive Infusion (CAI)**.

The current implementation uses five predefined infusion levels, layer-specific Boolean gate preferences, accuracy-based progression, rollback, and potentially dense-layer freezing.

I want to replace these discrete infusion levels with a **mathematically defined, continuous, self-adaptive controller** that automatically adjusts infusion strength based on:

1. Validation accuracy.
2. Effective Boolean-circuit complexity.
3. Model sensitivity to infusion.
4. Training progress.

The method should automatically adapt to different datasets and architectures. A robust model should be able to start with stronger infusion, while a sensitive model should start more gently.

**CRITICAL: DO NOT MODIFY, DELETE, RENAME, OR REWRITE ANY EXISTING SOURCE CODE, TRAINING SCRIPTS, CONFIGURATIONS, OR CHECKPOINT IMPLEMENTATIONS.**

This must be a separate implementation using new files and a new training entry point. The original Progressive Aggressive Infusion must remain fully functional and reproducible.

## 2. First: Inspect the existing implementation

Before writing code:

- Locate and study `ProgressiveAggressiveInfusion`, `apply_infusion_level()`, `CandidateLogicLayer`, the compact controller, candidate resampling, and dense-freezing logic.
- Identify how Conv1–Conv4 and Dense1–Dense2 are handled.
- Identify the original candidate-sampling distribution and the five infusion distributions.
- Confirm operation IDs and the exact meaning of convolutional AND/OR preference, dense identity/inversion/constant preference, and uniform sampling.
- Inspect training, validation, checkpointing, rollback, random-state management, exporting and gate analysis.
- Understand whether candidate replacement, freezing, or other operations are irreversible.

Do not assume the five-level policy describes every behavior of the existing trainer. Verify the actual implementation.

**Before implementing CAI, briefly report what you found and which new files you will create.**

## 3. Continuous layer-aware infusion

Define a continuous infusion strength:

\[
\alpha_t \in [0,1]
\]

For each layer \(\ell\), use:

\[
P_{\ell,t}(k)
=
(1-\alpha_t)P_{\ell}^{base}(k)
+
\alpha_t P_{\ell}^{target}(k)
\]

Where:

- \(P_{\ell}^{base}\): the original, non-infused candidate-sampling distribution.
- \(P_{\ell}^{target}\): the layer-specific target infusion distribution.
- \(k\): a Boolean gate operation.
- \(\alpha_t\): continuously controlled infusion strength.

At \(\alpha=0\), sampling must reproduce original non-infused behavior.

At \(\alpha=1\), sampling reaches the designated target preferences.

Intermediate values must produce valid, normalized probabilities.

### Preserve the existing layer-specific logic

For the initial CAI implementation:

- Conv1–Conv3: retain the existing AND/OR preference.
- Conv4: retain the original behavior.
- Dense layers: retain the existing preferences for uniform, identity, inversion, and constant candidates.
- Use the most aggressive existing infusion configuration as the initial target, but verify its precise meaning first.

Do not introduce arbitrary new gate preferences.

Implement continuous interpolation at the correct level of the existing candidate-resampling machinery. If the code uses separate operation-prior and candidate-family distributions, preserve that distinction rather than forcing an incorrect unified distribution.

## 4. Automatic adaptive control

The controller should automatically increase or decrease infusion strength based on feedback.

### A. Smooth validation accuracy

\[
\bar A_t = \beta\bar A_{t-1}+(1-\beta)A_t
\]

Define:

\[
m_t=\bar A_t-(A_{ref}-\delta)
\]

Where:

- \(A_{ref}\) is an appropriate validated reference accuracy from a stable checkpoint.
- \(\delta\) is the acceptable accuracy degradation.
- \(m_t\) is the accuracy safety margin.

Ensure the accuracy reference is not artificially lowered when performance deteriorates.

### B. Measure effective circuit complexity

Define normalized simplification progress:

\[
g_t = \frac{C_{t-1}-C_t}{C_{ref}}
\]

Use an appropriate, computationally affordable circuit-complexity measurement based on the existing exported hard Boolean network.

Prefer measurements such as:

- Reachable gates after dead-gate elimination.
- Constants and redundant operations after simplification.
- Effective two-input Boolean operations.
- Estimated storage needed for the simplified circuit.

If existing tools cannot compute these metrics reliably, implement a clearly documented proxy and distinguish it from measured FPGA BRAM/LUT utilization.

Do not launch full Vivado synthesis at every validation step.

Smooth the complexity signal where necessary to reduce noise.

### C. Continuous feedback controller

Implement the following as the initial candidate controller:

\[
s_t=\tanh\left(\frac{m_t}{T_A}\right)
\]

\[
h_t=\epsilon+(1-\epsilon)
\sigma\left(\frac{g_t-g_{min}}{T_C}\right)
\]

\[
u_{t+1}=
\operatorname{clip}_{[0,1]}
\left[
u_t
+\eta_{up}\max(0,s_t)h_t
-\eta_{down}\max(0,-s_t)
\right]
\]

Then smooth the applied infusion strength:

\[
\alpha_{t+1}=(1-\rho)\alpha_t+\rho u_{t+1}
\]

Where:

- \(\sigma\) is the sigmoid function.
- \(u_t\) is the controller's target strength.
- \(\alpha_t\) is the applied strength.
- \(\eta_{up}\) controls progression.
- \(\eta_{down}\) controls recovery.
- \(\rho\) controls smoothing.
- \(\epsilon\) permits limited exploration when measured complexity improvement is small.
- \(T_A,T_C\) control feedback sensitivity.

The desired behavior:

- Safe accuracy + meaningful simplification: increase infusion.
- Safe accuracy + little simplification: progress cautiously or hold.
- Accuracy deterioration: decrease infusion promptly.
- Severe regression: restore a safe checkpoint.

Use conservative defaults and document all control parameters.

Include safeguards against noisy validation measurements, oscillation, repeated unproductive resampling, and excessive simplification.

Do not assume this formula is optimal. Implement it transparently so it can be tested against simpler accuracy-only feedback.

## 5. Automatically calibrate initial infusion strength

This is particularly important.

**Do not force every model or dataset to start with the same infusion strength.**

Create a short optional calibration phase after an appropriate initial training warm-up.

The controller should estimate how sensitive the current model is to candidate infusion.

For example:

- Probe candidate infusion strengths within \([0,1]\).
- Evaluate changes in validation accuracy and effective circuit complexity.
- Select a conservative initial strength that meets the configured accuracy constraint.
- Start more aggressively when calibration indicates that the model tolerates infusion.
- Start gently when the model appears sensitive.

Prefer an adaptive search or bounded probing procedure rather than another fixed five-level schedule.

Calibration must:

- Use validation data, never test data.
- Use the same evaluation protocol for each candidate.
- Avoid permanently altering the original training state during probes.
- Save and restore model, optimizer, controller, freezing, and CPU/CUDA RNG states as necessary.
- Account for the effects of discrete gate resampling.
- Have configurable computational limits.
- Fall back safely to gentle infusion if calibration is inconclusive.

Clearly record calibration results and the selected initial \(\alpha\).

## 6. Handle discrete resampling and freezing correctly

Continuous probabilities do not imply continuous changes in the actual circuit.

Therefore:

- Retain the existing candidate-resampling mechanism wherever possible.
- Limit the amount of resampling per update if needed.
- Ensure changes in \(\alpha\) affect future sampling probabilities, not unexplained changes to already selected gates.
- Do not assume reducing \(\alpha\) reverses previous resampling or permanently frozen gates.
- Preserve safe checkpoint rollback.
- Check that restored optimizer, RNG, controller and freezer states are internally consistent.

If permanent freezing makes adaptive recovery unsafe, introduce a compatible policy in the new implementation rather than altering the old one.

## 7. Implement as a completely separate method

Create new files for:

- CAI configuration.
- Continuous infusion controller.
- Initial calibration.
- Circuit complexity measurement, if needed.
- A standalone CAI training entry point.
- CAI-specific checkpoint save/load functions.
- Unit and integration tests.
- A short README explaining how to run the method.

Reuse existing classes through imports and composition when possible.

**Never edit the existing Progressive Aggressive Infusion files.**

If integration requires changing an existing file, do not make that change. Instead, create a new adapter, wrapper, subclass, or independent entry point.

If some feature cannot be implemented safely without altering existing code, explain the blocker rather than silently modifying it.

## 8. Checkpointing and experiment tracking

A CAI checkpoint must preserve everything necessary for deterministic resume, including:

- Training step and epoch.
- Model and optimizer states.
- Existing compact-controller state.
- Continuous-controller state, including \(u_t\) and \(\alpha_t\).
- EMA values and feedback history.
- Calibration results.
- Circuit-complexity baseline and recent measurements.
- Dense-freezer state, if enabled.
- Safe-checkpoint information.
- CPU and CUDA RNG states.
- Training configuration and metrics.

Avoid conflicts with original checkpoints by using a distinct checkpoint kind and output directory.

Log at least:

- Training/validation accuracy and loss.
- Current and target infusion strengths.
- Layer-specific candidate probabilities.
- Effective circuit complexity.
- Gate-operation distribution.
- Infusion/resampling events.
- Rollback events.
- Selected calibration strength.

## 9. Testing and scientific comparisons

Add tests confirming:

1. \(\alpha=0\) reproduces original non-infused sampling.
2. \(\alpha=1\) reaches the intended target distribution.
3. All interpolated probabilities are non-negative and sum to one.
4. Accuracy deterioration reduces infusion strength.
5. Stable accuracy permits controlled progression.
6. The controller does not oscillate uncontrollably.
7. Calibration restores all states correctly.
8. Checkpoint resume is reproducible.
9. Original Progressive Aggressive Infusion remains untouched.

Make it straightforward to run matched experiments using:

- Original training, no infusion.
- Existing five-level Progressive Aggressive Infusion.
- Continuous Adaptive Infusion with accuracy feedback only.
- Continuous Adaptive Infusion with accuracy and complexity feedback.

These experiments should share architecture, dataset split, training budget, seed, and hardware evaluation pipeline.

Do not claim CAI improves accuracy or hardware efficiency before experiments demonstrate it.

## 10. Final deliverables

After implementation:

1. List every new file created.
2. Confirm that no existing files were modified.
3. Explain the final mathematical controller.
4. Explain how automatic initial calibration works.
5. Provide the exact commands to train and resume CAI.
6. Explain how to compare CAI with the original method fairly.
7. Report tests executed and any unresolved limitations.

### Important research objective

The intended contribution for an ICML paper is:

**A layer-aware, continuously adaptive Boolean-gate infusion method that automatically calibrates and regulates circuit-simplification pressure according to model sensitivity, validation performance, and effective circuit complexity.**

Prioritize scientific validity, reproducibility, modularity, and compatibility with the existing implementation.

**Do not change the previous training implementation under any circumstances.**
