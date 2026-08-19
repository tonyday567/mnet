{-# LANGUAGE NoImplicitPrelude #-}

-- | Oracles for the circuits-ecosystem rewrite of mnet.
--
--   1. Identity-training oracle: a 2-layer MLP trained on the identity
--      mapping must decrease loss.
--   2. Coalgebra oracle: the same training loop expressed as a
--      'Circuit.ChannelPoly.Coalgebra' must also decrease loss.
--   3. Gradient oracle: gradients produced by 'circuits-ad' via 'DiffP'
--      must agree with hand-rolled reference gradients on a fixed input.
module Main where

import Circuit.Mat.Dense (Matrix (..))
import Harpie.Array (arrayAs)
import Harpie.Array qualified as A
import Net (NetParams (..), netParamsFromArrays)
import NetCoalgebra
  ( forward,
    gradientsViaAD,
    mseLoss,
    referenceGradients,
    trainIdentity,
    trainNetCoalgebra,
  )
import System.Exit (exitFailure, exitSuccess)
import Prelude hiding (id, (.))

main :: IO ()
main = do
  -- Create small net: 4 → 8 → 4 (identity mapping)
  let w1Init = A.tabulate [8, 4] (const 1.0) :: A.Array Double
  let b1Init = A.tabulate [8] (const 0.0) :: A.Array Double
  let w2Init = A.tabulate [4, 8] (const 1.0) :: A.Array Double
  let b2Init = A.tabulate [4] (const 0.0) :: A.Array Double
  let params = netParamsFromArrays w1Init b1Init w2Init b2Init

  -- Input/target: all-ones vector
  let x = A.tabulate [4] (const 1.0) :: A.Array Double
  let target = A.tabulate [4] (const 1.0) :: A.Array Double

  -- Train for 500 steps
  let lr = 0.01
  let history = trainIdentity 500 lr params x target
  let losses = map fst history
      initialLoss = case losses of (l : _) -> l; [] -> 0
      finalLoss = case reverse losses of (l : _) -> l; [] -> 0

  putStrLn $ "Initial loss: " ++ show initialLoss
  putStrLn $ "Final loss:   " ++ show finalLoss
  putStrLn $ "Steps:        " ++ show (length history)

  -- Oracle 1: net loss decrease
  if finalLoss < initialLoss / 2
    then putStrLn "PASS: loss decreased substantially"
    else do
      putStrLn $ "FAIL: loss " ++ show initialLoss ++ " -> " ++ show finalLoss
      exitFailure

  -- Oracle 2: final loss is small
  if finalLoss < 0.1
    then putStrLn "PASS: final loss < 0.1"
    else do
      putStrLn $ "FAIL: final loss " ++ show finalLoss ++ " >= 0.1"
      exitFailure

  -- Oracle 3: forward pass produces correct shape
  let y = forward params x
  if A.size y == 4
    then putStrLn "PASS: output shape correct (4)"
    else do
      putStrLn $ "FAIL: output shape " ++ show (A.size y) ++ " /= 4"
      exitFailure

  ----------------------------------------------------------------------
  -- Coalgebra bridge oracle
  ----------------------------------------------------------------------
  do
    let coalHistory = trainNetCoalgebra params x target lr 500
        coalInitial = fst (mseLoss (forward params x) target)
        coalFinal = case coalHistory of
          [] -> coalInitial
          hs -> fst (mseLoss (forward (snd (last hs)) x) target)
    putStrLn $ "Coalgebra initial loss: " ++ show coalInitial
    putStrLn $ "Coalgebra final loss:   " ++ show coalFinal

    if coalFinal < coalInitial / 2
      then putStrLn "PASS: coalgebra loss decreased substantially"
      else do
        putStrLn $ "FAIL: coalgebra loss " ++ show coalInitial ++ " -> " ++ show coalFinal
        exitFailure

    if coalFinal < 0.1
      then putStrLn "PASS: coalgebra final loss < 0.1"
      else do
        putStrLn $ "FAIL: coalgebra final loss " ++ show coalFinal ++ " >= 0.1"
        exitFailure

  ----------------------------------------------------------------------
  -- Gradient oracle: circuits-ad vs hand-rolled reference
  ----------------------------------------------------------------------
  do
    let adGrad = gradientsViaAD params x target
        refGrad = referenceGradients params x target
        adFlat = flatten adGrad
        refFlat = flatten refGrad
        diffs = zipWith (-) adFlat refFlat
        relErrs = zipWith (\a r -> abs (a - r) / max 1e-12 (abs r)) adFlat refFlat
        maxRelErr = maximum relErrs
    putStrLn $ "Gradient vector length: " ++ show (length adFlat)
    putStrLn $ "Max absolute gradient diff: " ++ show (maximum (map abs diffs))
    putStrLn $ "Max relative gradient diff: " ++ show maxRelErr

    if maxRelErr < 1e-10
      then putStrLn "PASS: circuits-ad gradients match reference"
      else do
        putStrLn $ "FAIL: circuits-ad gradients diverge from reference (max rel err " ++ show maxRelErr ++ ")"
        exitFailure

  putStrLn "All oracles passed."
  exitSuccess

-- | Flatten a 'NetParams' bundle into a single list for comparison.
flatten :: NetParams Double -> [Double]
flatten p =
  concat
    [ arrayAs (unMatrix (w1 p)),
      arrayAs (b1 p),
      arrayAs (unMatrix (w2 p)),
      arrayAs (b2 p)
    ]
