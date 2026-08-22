{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE NoImplicitPrelude #-}

-- | Skip-Para neural network expressed as a polynomial coalgebra over the
-- circuits ecosystem.
--
-- * Linear maps use 'Circuit.Mat.Dense'.
-- * Gradients are produced via 'Circuit.Diff.Param.DiffP' (reverse mode).
-- * Parameter updates run through 'Circuit.Learn.Ephemeral.sgd' 'Progress'.
-- * The whole training step is packaged as a 'Circuit.System.Coalgebra'
--   whose output direction is a learning-rate scalar.
--
-- A 'These'-based boundary type is provided for future batch scheduling;
-- batching via 'Circuit.Tensor' is left as a TODO stub.
module NetCoalgebra
  ( -- * Parameter bundle (re-exported from 'Net')
    Net.NetParams (..),
    Net.netParamsFromArrays,

    -- * Forward pass
    Net.forward,
    Net.mseLoss,

    -- * Training state
    NetState (..),
    initNetState,

    -- * Training step and coalgebra
    trainStep,
    netCoalgebra,
    trainNetCoalgebra,
    trainIdentity,

    -- * Gradient oracle helpers
    gradientsViaAD,
    referenceGradients,

    -- * Boundary type
    Net.Boundary,

    -- * Batch scheduling stub
    batchSchedule,
  )
where

import Circuit.System (Coalgebra (..))
import Circuit.Diff.Param (DiffP (..), runDiffP)
import Circuit.Learn.Ephemeral (Progress (..), sgd)
import Circuit.Mat.Dense (Matrix (..), matTimes, matVec)
import Circuit.Poly (Eval (..), Mono, lens)
import Data.Vector.Unboxed qualified as VU
import Harpie.Array (Array, arrayAs)
import Harpie.Array qualified as A
import Net qualified
import NumHask.Algebra.Additive (zero)
import NumHask.Algebra.Multiplicative (one)
import Prelude hiding (id, (.))

-- | Flatten all parameters into a single vector.
flattenParams :: Net.NetParams Double -> [Double]
flattenParams p =
  concat
    [ arrayAs (unMatrix (Net.w1 p)),
      arrayAs (Net.b1 p),
      arrayAs (unMatrix (Net.w2 p)),
      arrayAs (Net.b2 p)
    ]

-- | Reshape a flat parameter vector according to the shapes of a reference
-- parameter bundle.
reshapeParams :: [Double] -> Net.NetParams Double -> Net.NetParams Double
reshapeParams flat p =
  let (w1Flat, rest1) = splitAt (A.size (unMatrix (Net.w1 p))) flat
      (b1Flat, rest2) = splitAt (A.size (Net.b1 p)) rest1
      (w2Flat, rest3) = splitAt (A.size (unMatrix (Net.w2 p))) rest2
      (b2Flat, _) = splitAt (A.size (Net.b2 p)) rest3
      mkArr a vals = A.array (VU.toList (A.shape a)) vals
   in Net.NetParams
        { Net.w1 = Matrix (mkArr (unMatrix (Net.w1 p)) w1Flat),
          Net.b1 = mkArr (Net.b1 p) b1Flat,
          Net.w2 = Matrix (mkArr (unMatrix (Net.w2 p)) w2Flat),
          Net.b2 = mkArr (Net.b2 p) b2Flat
        }

-- | Row count of a dense matrix.
rows :: Matrix a -> Int
rows (Matrix a) = case VU.toList (A.shape a) of [r, _] -> r; _ -> 0

-- | Matrix–vector product, returning an 'Array'.
matVecArr :: Matrix Double -> Array Double -> Array Double
matVecArr m v = A.array [rows m] (matVec m (arrayAs v))

-- | Dense matrix transpose.
transposeMatrix :: Matrix Double -> Matrix Double
transposeMatrix (Matrix a) =
  case VU.toList (A.shape a) of
    [r, c] ->
      Matrix
        ( A.tabulate
            [c, r]
            ( \case
                [i, j] -> a A.! [j, i]
                _ -> error "transposeMatrix: expected 2-element index"
            )
        )
    _ -> error "transposeMatrix: expected rank-2 matrix"

-- | Outer product of two vectors as a dense matrix.
outerProduct :: Array Double -> Array Double -> Matrix Double
outerProduct dy x =
  let r = A.size dy
      c = A.size x
      dyVals = arrayAs dy :: [Double]
      xVals = arrayAs x :: [Double]
      dyMat = Matrix (A.array [r, 1] dyVals)
      xTMat = Matrix (A.array [1, c] xVals)
   in matTimes dyMat xTMat

-- | Rectified linear unit.
reluArr :: Array Double -> Array Double
reluArr = fmap (\x -> max x zero)

-- | ReLU derivative.
reluGradArr :: Array Double -> Array Double
reluGradArr = fmap (\x -> if x > zero then one else zero)

-- | Full network as a parameterised differentiable arrow.
--
-- The backward pass closes over the forward intermediates and returns the
-- input cotangent together with a full 'NetParams' gradient.  This is the
-- honest 'circuits-ad' reverse-mode shape for the network.
modelDiffP :: DiffP (Net.NetParams Double) (Array Double) (Array Double)
modelDiffP = DiffP $ \p x ->
  let a1 = matVecArr (Net.w1 p) x
      z1 = A.zipWith (+) a1 (Net.b1 p)
      h1 = reluArr z1
      a2 = matVecArr (Net.w2 p) h1
      y = A.zipWith (+) a2 (Net.b2 p)
   in ( y,
        \dy ->
          let dz2 = dy
              da2 = dz2
              dh1 = matVecArr (transposeMatrix (Net.w2 p)) da2
              dz1 = A.zipWith (*) dh1 (reluGradArr z1)
              da1 = dz1
              dx = matVecArr (transposeMatrix (Net.w1 p)) da1
              dw1 = outerProduct da1 x
              db1 = da1
              dw2 = outerProduct da2 h1
              db2 = dz2
           in ( dx,
                Net.NetParams
                  { Net.w1 = dw1,
                    Net.b1 = db1,
                    Net.w2 = dw2,
                    Net.b2 = db2
                  }
              )
      )

-- | Training state: the parameter bundle.  The optimiser is stateless
-- plain SGD, represented as a 'Circuit.Learn.Ephemeral.Progress' that is
-- instantiated at step time with the current gradient and learning rate.
newtype NetState = NetState
  { netParams :: Net.NetParams Double
  }

-- | Initial training state.
initNetState :: Net.NetParams Double -> NetState
initNetState = NetState

-- | One training step.
--
-- Returns the prediction and a function that consumes the learning-rate
-- direction and produces the next state.
trainStep ::
  NetState ->
  (Array Double, Array Double) ->
  (Array Double, Double -> NetState)
trainStep st (x, target) =
  let p = netParams st
      (y, back) = runDiffP modelDiffP p x
      (_, grad) = Net.mseLoss y target
      (_dx, gradParams) = back grad
      gradFlat = flattenParams gradParams
      apply lr =
        let flat = flattenParams p
            -- 'Circuit.Learn.Ephemeral.sgd' expects a gradient function; we
            -- supply the precomputed gradient for this fixed example.
            progress = sgd lr (\_ _ -> gradFlat)
            flat' = step progress (x, target) flat
         in NetState (reshapeParams flat' p)
   in (y, apply)

-- | Polynomial coalgebra view of the network.
--
-- * State: 'NetState'.
-- * Input position: @(input, target)@; no input direction.
-- * Output position: prediction; output direction: learning rate.
netCoalgebra ::
  Coalgebra
    NetState
    (Mono () (Array Double, Array Double))
    (Mono Double (Array Double))
netCoalgebra =
  Coalgebra
    { act =
        \st ->
          lens
            (\(x, _) -> Net.forward (netParams st) x)
            (\_ _ -> ()),
      upd =
        \st (EP (EK (x, target), _)) ->
          let (y, apply) = trainStep st (x, target)
           in EP (EK y, EE apply)
    }

-- | Train the coalgebra for @n@ steps on a fixed @(x, target)@ pair.
trainNetCoalgebra ::
  Net.NetParams Double ->
  Array Double ->
  Array Double ->
  Double ->
  Int ->
  [(Double, Net.NetParams Double)]
trainNetCoalgebra p0 x target lr n = trainIdentity n lr p0 x target

-- | Identity-mapping training oracle: train for @n@ steps and return the
-- loss before each update together with the parameters after each update.
trainIdentity ::
  Int ->
  Double ->
  Net.NetParams Double ->
  Array Double ->
  Array Double ->
  [(Double, Net.NetParams Double)]
trainIdentity n lr p0 x target = go (initNetState p0) n
  where
    go _ 0 = []
    go st k =
      let y = Net.forward (netParams st) x
          (loss, _) = Net.mseLoss y target
          (_, apply) = trainStep st (x, target)
          st' = apply lr
       in (loss, netParams st') : go st' (k - 1)

-- | Compute parameter gradients via the 'circuits-ad' 'DiffP' model.
gradientsViaAD ::
  Net.NetParams Double ->
  Array Double ->
  Array Double ->
  Net.NetParams Double
gradientsViaAD p x target =
  let (y, back) = runDiffP modelDiffP p x
      (_, grad) = Net.mseLoss y target
      (_, gradParams) = back grad
   in gradParams

-- | Hand-rolled reference gradients for the 2-layer network, kept to verify
-- the 'circuits-ad' / 'DiffP' gradients in the oracle.
--
-- This is the old skip-Para chain rule, written directly against the same
-- 'Circuit.Mat.Dense' primitives used by the rest of the module.
referenceGradients ::
  Net.NetParams Double ->
  Array Double ->
  Array Double ->
  Net.NetParams Double
referenceGradients p x target =
  let a1 = matVecArr (Net.w1 p) x
      z1 = A.zipWith (+) a1 (Net.b1 p)
      h1 = reluArr z1
      a2 = matVecArr (Net.w2 p) h1
      y = A.zipWith (+) a2 (Net.b2 p)
      dOut = snd (Net.mseLoss y target)
      db2 = dOut
      dw2 = outerProduct dOut h1
      da2 = dOut
      dh1 = matVecArr (transposeMatrix (Net.w2 p)) da2
      dz1 = A.zipWith (*) dh1 (reluGradArr z1)
      db1 = dz1
      da1 = dz1
      dw1 = outerProduct da1 x
   in Net.NetParams dw1 db1 dw2 db2

-- | Batch scheduling stub.  The intended implementation uses
-- 'Circuit.Tensor' combinators (broadcast / telecast / indexWindows) to
-- turn single-example layers into batch layers and fold gradients across
-- the batch dimension.
batchSchedule :: a
batchSchedule = error "batchSchedule: not yet implemented"
