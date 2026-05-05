# mnet

A modular neural network library based on parameterised morphisms and bidirectional Store composition.

based on:

https://cybercat.institute/2024/04/15/neural-network-first-principles/

## Overview

`mnet` provides a clean, composable approach to neural networks using:

- **Para** - Parameterised morphisms for functions with shared parameter environments
- **LensS** - Van Laarhoven-style lenses using the Store comonad
- **Net** - Neural network layers and training infrastructure

## Building

```bash
cabal build
```

## Modules

### Para
Parameterised functions: `Para p a b = (p, a) → b`

Both forward and backward passes share the same parameter environment.

### LensS
Lenses as Store-wrapped functions, composable via category.

### Net
Neural network layers (linear, bias, ReLU) with forward and backward passes using Store.

## Version

0.1.0.0 - Initial release

## License

BSD-3-Clause
