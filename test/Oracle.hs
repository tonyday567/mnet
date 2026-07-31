{-# LANGUAGE NoImplicitPrelude #-}

-- | Oracle: verify the skip-Para neural net trains correctly.
--
-- Trains a 2-layer MLP on the identity mapping.  Checks:
--   1. Forward pass produces correct output shape
--   2. Loss decreases substantially over training
--   3. After 500 steps, loss is below threshold
module Main where

import Harpie.Array qualified as A
import NetCoalgebra
import NetCoalgebra.Poly (trainNetCoalgebra)
import System.Exit (exitFailure, exitSuccess)
import Prelude hiding (id, (.))

main :: IO ()
main = do
  -- Create small net: 4 → 8 → 4 (identity mapping)
  let w1Init = A.tabulate [8, 4] (const 1.0) :: A.Array Double
  let b1Init = A.tabulate [8] (const 0.0) :: A.Array Double
  let w2Init = A.tabulate [4, 8] (const 1.0) :: A.Array Double
  let b2Init = A.tabulate [4] (const 0.0) :: A.Array Double
  let params = NetParams w1Init b1Init w2Init b2Init

  -- Input/target: all-ones vector
  let x = A.tabulate [4] (const 1.0) :: A.Array Double
  let target = A.tabulate [4] (const 1.0) :: A.Array Double

  -- Train for 500 steps
  let lr = 0.01
  let history = trainIdentity 500 lr params x target
  let losses = map fst history

  putStrLn $ "Initial loss: " ++ show (head losses)
  putStrLn $ "Final loss:   " ++ show (last losses)
  putStrLn $ "Steps:        " ++ show (length history)

  -- Oracle 1: net loss decrease
  if last losses < head losses / 2
    then putStrLn "PASS: loss decreased substantially"
    else do
      putStrLn $ "FAIL: loss " ++ show (head losses) ++ " -> " ++ show (last losses)
      exitFailure

  -- Oracle 2: final loss is small
  if last losses < 0.1
    then putStrLn "PASS: final loss < 0.1"
    else do
      putStrLn $ "FAIL: final loss " ++ show (last losses) ++ " >= 0.1"
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

  putStrLn "All oracles passed."
  exitSuccess
