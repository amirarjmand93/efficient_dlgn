## DLGN Progress and Remaining Challenges

The current DLGN has improved through several targeted changes while preserving its rank-2 Boolean/LUT structure and hardware-friendly design. The most successful improvements so far have been the **multi-bit input encoding**, especially quantile-based and hybrid Q+S representations, and **CIFAR random crop + horizontal-flip augmentation**, which produced a clear accuracy improvement. Class-routed dense layers were also explored, although their benefit appears smaller and less consistent.

The network still follows the same core pipeline:

**binary encoder → LogicConv layers → LogicDense/class-routed layers → GroupSum classification**

Several alternatives—including attention, extra convolutional depth, larger widths, random wire erasure, random erasing, and modified dense structures—did not provide reliable gains.

### LogicConv remains an important research direction

The **LogicConv layers are not fully solved and may still be a major bottleneck**. In the current implementation, every logic tree uses a small fixed subset of input wires. In the first convolution, this can strongly restrict interaction between different encoded channels such as RGB, quantile, and sine bits.

Simply increasing the channel group size did not improve performance, and the first `channel-balanced` experiment also appeared too aggressive: mixing many channels within only eight available tree leaves reduces the amount of spatial information seen from each channel.

A more promising direction is therefore to redesign LogicConv connectivity so that each kernel jointly preserves:

- **spatial structure** inside the receptive field,
- **cross-channel interaction** between encoded features,
- **balanced channel usage** across kernels,
- and the same small rank-2 logic-tree/hardware cost.

For example, a kernel could use **two complementary channels across four shared spatial locations**, rather than eight unrelated channels from eight individual pixels.

### Main remaining challenges

The highest-priority areas are therefore:

1. **LogicConv connectivity** — improve channel mixing without destroying spatial structure.
2. **Generalization / overfitting** — reduce the gap between training and test performance.
3. **Fixed topology** — investigate whether better or partially learned/stochastic connectivity can outperform one randomly fixed wiring.
4. **Relaxed-to-discrete gap** — improve conversion from the differentiable training model to the final Boolean network.
5. **Dense head efficiency** — determine whether the large LogicDense section can be simplified or regularized without reducing accuracy.

The most promising current direction is therefore not simply making the DLGN wider or deeper, but improving **how information is connected and composed inside the existing logic gates** while keeping the final network compact and hardware-friendly.