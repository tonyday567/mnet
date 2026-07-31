{-# LANGUAGE NoImplicitPrelude #-}

-- | Skip-Para neural network: explicit state threading, no Para wrapper.
--
-- This is the coalgebra alternative to the Para-based @Net@ module.
-- Instead of @Para p a b@, every layer takes state explicitly:
--
-- @
-- layer :: state -> input -> output
-- @
--
-- The backward pass is separate from the forward pass.  The optimiser
-- is external — it reads gradients and updates state.  This matches the
-- Spivak coalgebra pattern: state determines the morphism; backward flow
-- updates the state; the optimiser is an external controller.
module NetCoalgebra where

import Harpie.Array qualified as A
import Prelude

-- | Network parameters (weights and biases).
data NetParams a = NetParams
  { w1 :: A.Array a,
    b1 :: A.Array a,
    w2 :: A.Array a,
    b2 :: A.Array a
  }
  deriving (Show)

-- * Forward pass — each layer: state → input → output

linear1Fwd :: (Num a) => NetParams a -> A.Array a -> A.Array a
linear1Fwd p = A.mult (w1 p)

bias1Fwd :: (Num a) => NetParams a -> A.Array a -> A.Array a
bias1Fwd p x = x + b1 p

reluFwd :: (Ord a, Num a) => A.Array a -> A.Array a
reluFwd = fmap (max 0)

linear2Fwd :: (Num a) => NetParams a -> A.Array a -> A.Array a
linear2Fwd p = A.mult (w2 p)

bias2Fwd :: (Num a) => NetParams a -> A.Array a -> A.Array a
bias2Fwd p x = x + b2 p

-- | Full forward pass: linear1 → bias1 → relu → linear2 → bias2.
forward :: (Num a, Ord a) => NetParams a -> A.Array a -> A.Array a
forward p = bias2Fwd p . linear2Fwd p . reluFwd . bias1Fwd p . linear1Fwd p

-- * Backward pass — each layer: state → input → outputGrad → inputGrad

-- | Gradient of linear layer w.r.t. input: @W^T · dy@.
linear1Bwd :: (Num a) => NetParams a -> A.Array a -> A.Array a -> A.Array a
linear1Bwd p _x = A.mult (A.transpose (w1 p))

bias1Bwd :: A.Array a -> A.Array a
bias1Bwd = id

-- | ReLU backward: gradient flows through where input > 0.
reluBwd :: (Ord a, Num a) => A.Array a -> A.Array a -> A.Array a
reluBwd = A.zipWith (\xi dyi -> if xi > 0 then dyi else 0)

linear2Bwd :: (Num a) => NetParams a -> A.Array a -> A.Array a -> A.Array a
linear2Bwd p _x = A.mult (A.transpose (w2 p))

bias2Bwd :: A.Array a -> A.Array a
bias2Bwd = id

-- | Full backward pass: reverse order of forward.
backward ::
  (Ord a, Fractional a) =>
  NetParams a ->
  A.Array a ->
  A.Array a ->
  A.Array a
backward p x dOut =
  let -- Save intermediate values (forward pass)
      a1 = linear1Fwd p x
      z1 = bias1Fwd p a1
      h1 = reluFwd z1
      -- Backward pass (reverse order)
      dz2 = bias2Bwd dOut
      da2 = linear2Bwd p h1 dz2
      dh1 = reluBwd z1 da2
      dz1 = bias1Bwd dh1
      da1 = linear1Bwd p x dz1
   in da1

-- * Parameter gradients — how to update each parameter

-- | Gradient of linear1 weights: @dy · x^T@.
linear1WGrad ::
  (Num a) =>
  NetParams a ->
  A.Array a ->
  A.Array a ->
  A.Array a
linear1WGrad _p x dy = A.expand (*) dy x

-- | Gradient of bias1: same as dy.
bias1Grad :: A.Array a -> A.Array a
bias1Grad = id

-- | Gradient of linear2 weights.
linear2WGrad ::
  (Num a) =>
  NetParams a ->
  A.Array a ->
  A.Array a ->
  A.Array a
linear2WGrad _p h1 dy = A.expand (*) dy h1

-- | Gradient of bias2.
bias2Grad :: A.Array a -> A.Array a
bias2Grad = id

-- * Optimiser — external controller, reads gradients, updates state

-- | Apply gradient descent update to all parameters.
applyUpdate ::
  (Num a) =>
  a ->
  NetParams a ->
  A.Array a ->
  A.Array a ->
  A.Array a ->
  A.Array a ->
  NetParams a
applyUpdate lr p dw1 db1 dw2 db2 =
  p
    { w1 = w1 p - fmap (lr *) dw1,
      b1 = b1 p - fmap (lr *) db1,
      w2 = w2 p - fmap (lr *) dw2,
      b2 = b2 p - fmap (lr *) db2
    }

-- | MSE loss: @(1/n) Σ(y - target)²@.
mseLoss ::
  (Fractional a) =>
  A.Array a ->
  A.Array a ->
  (a, A.Array a)
mseLoss y target =
  let diff = y - target
      n = fromIntegral (A.size y)
      loss = sum (fmap (^ (2 :: Int)) diff) / n
      grad = fmap (* (2 / n)) diff
   in (loss, grad)

-- | One training step: forward → loss → backward → update.
step ::
  (Ord a, Fractional a) =>
  a ->
  NetParams a ->
  A.Array a ->
  A.Array a ->
  (a, NetParams a)
step lr p x target =
  let -- Forward pass with intermediate values
      a1 = linear1Fwd p x
      z1 = bias1Fwd p a1
      h1 = reluFwd z1
      a2 = linear2Fwd p h1
      y = bias2Fwd p a2
      -- Loss + output gradient
      (loss, dOut) = mseLoss y target
      -- Backward pass (reverse order)
      db2 = bias2Grad dOut
      dw2 = linear2WGrad p h1 dOut
      da2 = linear2Bwd p h1 dOut
      dh1 = reluBwd z1 da2
      db1 = bias1Grad dh1
      dw1 = linear1WGrad p x dh1
      -- Update
      p' = applyUpdate lr p dw1 db1 dw2 db2
   in (loss, p')

-- * Oracle — verify the skip-Para net trains

-- | Train for N steps on identity mapping, return final loss.
-- Oracle: loss should decrease monotonically.
trainIdentity ::
  (Ord a, Fractional a) =>
  Int ->
  a ->
  NetParams a ->
  A.Array a ->
  A.Array a ->
  [(a, NetParams a)]
trainIdentity 0 _ _p _ _ = []
trainIdentity n lr p x target =
  let (loss, p') = step lr p x target
   in (loss, p') : trainIdentity (n - 1) lr p' x target
