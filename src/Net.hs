{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE UnicodeSyntax #-}

-- | Forward-only neural-network layer definitions built on the circuits
-- ecosystem.
--
-- * 'NetParams' stores weights as 'Circuit.Mat.Dense.Matrix' and biases as
--   'Harpie.Array' vectors.
-- * Linear layers use 'Circuit.Mat.Dense.matVec'.
-- * The full model is a 'Circuit.Learn.Para' composition, so inference
--   reuses the same parameter threading as the rest of circuits-learn.
module Net
  ( -- * Parameter bundle
    NetParams (..),
    netParamsFromArrays,

    -- * Layer primitives
    linear1,
    bias1,
    relu1,
    linear2,
    bias2,

    -- * Model
    model,
    forward,

    -- * Loss
    mseLoss,

    -- * Boundary type
    Boundary,
  )
where

import Circuit.Learn.Para (Para (..), runPara)
import Circuit.Mat.Dense (Matrix (..), matVec)
import Control.Category (Category (..))
import Data.These (These (..))
import Data.Vector.Unboxed qualified as VU
import Harpie.Array (Array)
import Harpie.Array qualified as A
import Harpie.Array (arrayAs)
import NumHask.Algebra.Additive (Additive, sum, zero)
import NumHask.Algebra.Multiplicative (Multiplicative)
import Prelude hiding (id, (.), sum)

-- | A 'These'-based boundary distinguishes inference-only, gradient-only,
-- and combined inference-with-gradient traffic at a layer boundary.
--
-- * 'This' prediction — inference only, no gradient.
-- * 'That' gradient — training signal only, no prediction.
-- * 'These' prediction gradient — training with prediction (common
--   supervised case).
type Boundary a = These a a

-- | Network parameters.  Weights are stored as dense matrices so that
-- 'Circuit.Mat.Dense' can be used for the linear maps; biases and
-- activations remain as 'Array' vectors.
data NetParams a = NetParams
  { w1 :: Matrix a,
    b1 :: Array a,
    w2 :: Matrix a,
    b2 :: Array a
  }
  deriving (Eq, Show)

-- | Build 'NetParams' from plain 'Array' weights and biases.
netParamsFromArrays ::
  Array a -> Array a -> Array a -> Array a -> NetParams a
netParamsFromArrays w1a b1a w2a b2a =
  NetParams (Matrix w1a) b1a (Matrix w2a) b2a

-- | Row count of a dense matrix.
rows :: Matrix a -> Int
rows (Matrix a) = case VU.toList (A.shape a) of [r, _] -> r; _ -> 0

-- | Matrix–vector product, returning an 'Array'.
matVecArr ::
  (Additive a, Multiplicative a) => Matrix a -> Array a -> Array a
matVecArr m v = A.array [rows m] (matVec m (arrayAs v))

-- | Rectified linear unit.
reluArr :: (Ord a, Additive a) => Array a -> Array a
reluArr = fmap (\x -> if x > zero then x else zero)

-- | First linear layer.
linear1 ::
  (Additive a, Multiplicative a) => Para (NetParams a) (Array a) (Array a)
linear1 = Para $ \(p, x) -> matVecArr (w1 p) x

-- | First bias layer.
bias1 :: (Num a) => Para (NetParams a) (Array a) (Array a)
bias1 = Para $ \(p, x) -> A.zipWith (+) x (b1 p)

-- | ReLU activation.
relu1 :: (Ord a, Additive a) => Para (NetParams a) (Array a) (Array a)
relu1 = Para $ \(_, x) -> reluArr x

-- | Second linear layer.
linear2 ::
  (Additive a, Multiplicative a) => Para (NetParams a) (Array a) (Array a)
linear2 = Para $ \(p, x) -> matVecArr (w2 p) x

-- | Second bias layer.
bias2 :: (Num a) => Para (NetParams a) (Array a) (Array a)
bias2 = Para $ \(p, x) -> A.zipWith (+) x (b2 p)

-- | Full 2-layer MLP: linear1 → bias1 → relu → linear2 → bias2.
model ::
  (Ord a, Num a, Additive a, Multiplicative a) =>
  Para (NetParams a) (Array a) (Array a)
model = bias2 . linear2 . relu1 . bias1 . linear1

-- | Run the model with explicit parameters.
forward ::
  (Ord a, Num a, Additive a, Multiplicative a) =>
  NetParams a ->
  Array a ->
  Array a
forward = runPara model

-- | Mean-squared-error loss and its gradient w.r.t. the prediction.
--
-- @L = (1/n) Σ (y - target)²@, @dL/dy = (2/n)(y - target)@.
mseLoss ::
  (Fractional a, Additive a) =>
  Array a ->
  Array a ->
  (a, Array a)
mseLoss y target =
  let diff = A.zipWith (-) y target
      n = fromIntegral (A.size y)
      loss = sum (fmap (\x -> x * x) diff) / n
      grad = fmap (* (2 / n)) diff
   in (loss, grad)
