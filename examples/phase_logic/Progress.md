# Boolean Wire Erasure Experiment Report

## Method

A **Boolean Wire Erasure** regularizer was added after the first `LogicDense` layer. During training, each activation has a probability of \(p=0.02\) of being replaced by **0.5**, representing an unknown Boolean state. The erasure is disabled during relaxed evaluation, discrete evaluation, and export, so it adds no inference cost.

## Experimental Setup

The original model configuration was kept unchanged:

* Model: `build_model_cifar10_smallmodel`
* Encoding: Q2 + S2 hybrid
* `k_num = 36`
* `lut_rank = 2`
* Parameters: 375,984
* Learning rate: 0.02, constant
* End temperature: 0.1
* Iterations: 57,000
* Wire erasure probability: \(p=0.02\)

## Preliminary Results

Up to step 41,000:

* Train relaxed accuracy: **69.42%**
* Test relaxed accuracy: **65.85%**
* Test discrete accuracy: **62.51%**
* Best discrete accuracy so far: **62.58%** at step 26,500

The previous baseline without wire erasure reached approximately **63.54%** best discrete accuracy.

## Interpretation

Wire erasure appears to act as intended as a regularizer by limiting training accuracy while maintaining relatively strong test relaxed accuracy. However, it has **not yet improved discrete test accuracy** over the baseline. The remaining performance limitation appears to be mainly the **relaxed-to-discrete gap**, rather than conventional overfitting.

## Next Step

Complete the current run to 57,000 iterations. If the final result remains below the baseline, test a weaker erasure probability such as:

$$
p=0.01
$$

rather than increasing the regularization strength.
