# mnet

A small neural-network library rebuilt as a client of the circuits ecosystem.

based on:

https://cybercat.institute/2024/04/15/neural-network-first-principles/

## Overview

`mnet` now delegates its building blocks to the circuits ecosystem instead of
rolling them by hand:

- **circuits-mat** — dense matrix/vector operations (`Circuit.Mat.Dense`).
- **circuits-ad** — reverse-mode automatic differentiation via
  `Circuit.AD.Param.DiffP`.
- **circuits-learn** — optimisers expressed as `Circuit.Process` machines.
- **Circuit.Poly / Circuit.ChannelPoly** — polynomial-coalgebra view of a
  training step.
- **Data.These** — boundary type distinguishing inference-only, gradient-only,
  and combined inference-with-gradient traffic.

The old hand-rolled `LensS`, `BackPass`, and `NetCoalgebra.Poly` modules have
been removed.

## Building

```bash
cabal build
```

## Modules

### Net

Forward-only layer definitions for a 2-layer MLP:

- `NetParams` stores weights as `Circuit.Mat.Dense.Matrix` and biases as
  `Harpie.Array` vectors.
- `linear1`, `bias1`, `relu1`, `linear2`, `bias2` are `Circuit.Learn.Para`
  morphisms.
- `model` composes the layers; `forward` runs inference.
- `mseLoss` provides mean-squared-error loss and output gradient.
- `Boundary` is a `These`-based type for inference/gradient boundaries.

### NetCoalgebra

The same network expressed as a polynomial coalgebra:

- `modelDiffP` is the network as a `DiffP` morphism; reverse-mode gradients
  are computed automatically.
- `NetState` pairs parameters with one `Process` per scalar parameter.
- `trainStep` returns a prediction and a learning-rate-consuming next-state
  function.
- `netCoalgebra` exposes the training step as a
  `Coalgebra NetState (Mono () (input, target)) (Mono Double prediction)`.
- `trainIdentity` / `trainNetCoalgebra` run the identity-mapping oracle.
- `referenceGradients` provides hand-rolled gradients for the gradient oracle.

## Oracles

```bash
cabal test oracle
```

The test suite checks:

1. `trainIdentity` decreases loss on the 4 → 8 → 4 identity task.
2. `trainNetCoalgebra` (the coalgebra view) also decreases loss.
3. `circuits-ad` gradients agree with the hand-rolled reference gradients.

## Version

0.1.0.0 — circuits-ecosystem rewrite.

## License

BSD-3-Clause
