{-# LANGUAGE DataKinds #-}

-- | Bridge from mnet's skip-Para neural network to 'Circuit.Poly.Coalgebra'.
--
-- mnet's 'NetCoalgebra.NetParams' forward pass is expressed as a polynomial
-- coalgebra: the parameters are the state, the input position is the pair
-- @(input, target)@, the output position is the prediction, and the output
-- direction is the learning rate supplied by the external optimiser.
module NetCoalgebra.Poly
  ( netCoalgebra,
    trainNetCoalgebra,
  )
where

import Circuit.Poly
  ( Eval (..),
    Mono,
    lens,
  )
import Circuit.ChannelPoly (Coalgebra (..))
import Harpie.Array (Array)
import NetCoalgebra
  ( NetParams (..),
    applyUpdate,
    bias1Fwd,
    bias1Grad,
    bias2Grad,
    forward,
    linear1Fwd,
    linear1WGrad,
    linear2Bwd,
    linear2WGrad,
    mseLoss,
    reluBwd,
    reluFwd,
  )

-- | A single neural-network layer/view as a coalgebra.
--
-- * State:      the network parameters @NetParams Double@.
-- * Input:      position @(input, target)@, no backward direction.
-- * Output:     position is the prediction @y@; direction is the learning
--               rate supplied by the external controller/optimiser.
-- * Dynamics:   the output direction @lr@ updates the parameters by gradient
--               descent.
netCoalgebra ::
  Coalgebra
    (NetParams Double)
    (Mono () (Array Double, Array Double))
    (Mono Double (Array Double))
netCoalgebra =
  Coalgebra
    { act =
        \params ->
          lens
            (\(x, _) -> forward params x)
            (\(_ :: (Array Double, Array Double)) (_ :: Double) -> ()),
      upd =
        \params (EP (EK (x, target), _)) ->
          let y = forward params x
              (_loss, dOut) = mseLoss y target
              -- Forward intermediates needed for parameter gradients.
              a1 = linear1Fwd params x
              z1 = bias1Fwd params a1
              h1 = reluFwd z1
              -- Parameter gradients (reverse order).
              db2 = bias2Grad dOut
              dw2 = linear2WGrad params h1 dOut
              da2 = linear2Bwd params h1 dOut
              dh1 = reluBwd z1 da2
              db1 = bias1Grad dh1
              dw1 = linear1WGrad params x dh1
              nextParams lr = applyUpdate lr params dw1 db1 dw2 db2
           in EP (EK y, EE nextParams)
    }

-- | Train the coalgebra for @n@ steps on a fixed @(x, target)@ pair.
--
-- Returns the loss *before* each update and the parameters *after* each
-- update, matching mnet's 'NetCoalgebra.trainIdentity' oracle.
trainNetCoalgebra ::
  NetParams Double ->
  Array Double ->
  Array Double ->
  Double ->
  Int ->
  [(Double, NetParams Double)]
trainNetCoalgebra params0 x target lr = go params0
  where
    go _ 0 = []
    go params n =
      let y = forward params x
          (loss, _) = mseLoss y target
          EP (EK _, EE nextParams) =
            upd netCoalgebra params (EP (EK (x, target), EE (\() -> params)))
          params' = nextParams lr
       in (loss, params') : go params' (n - 1)
