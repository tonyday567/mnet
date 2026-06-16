{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoImplicitPrelude #-}

module Net where

import Control.Category (Category (..))
import Harpie.Array qualified as A
import LensS (LensS (..), Store (..), getS, mkLensS, setS)
import Para (Para (..), runPara)
import Prelude hiding (id, (.))

data NetParams a = NetParams
  { w1 :: A.Array a,
    b1 :: A.Array a,
    w2 :: A.Array a,
    b2 :: A.Array a
  }

lensW1 :: LensS (NetParams a) (A.Array a)
lensW1 = mkLensS w1 (\p w -> p {w1 = w})

lensB1 :: LensS (NetParams a) (A.Array a)
lensB1 = mkLensS b1 (\p b -> p {b1 = b})

lensW2 :: LensS (NetParams a) (A.Array a)
lensW2 = mkLensS w2 (\p w -> p {w2 = w})

lensB2 :: LensS (NetParams a) (A.Array a)
lensB2 = mkLensS b2 (\p b -> p {b2 = b})

linear1 :: (Num a) => Para (NetParams a) (A.Array a) (A.Array a)
linear1 = Para $ \(p, x) -> A.mult (w1 p) x

bias1 :: (Num a) => Para (NetParams a) (A.Array a) (A.Array a)
bias1 = Para $ \(p, x) -> x + b1 p

relu1 :: (Ord a, Num a) => Para (NetParams a) (A.Array a) (A.Array a)
relu1 = Para $ \(_, x) -> fmap (max 0) x

linear2 :: (Num a) => Para (NetParams a) (A.Array a) (A.Array a)
linear2 = Para $ \(p, x) -> A.mult (w2 p) x

bias2 :: (Num a) => Para (NetParams a) (A.Array a) (A.Array a)
bias2 = Para $ \(p, x) -> x + b2 p

model :: (Num a, Ord a) => Para (NetParams a) (A.Array a) (A.Array a)
model = bias2 . linear2 . relu1 . bias1 . linear1

forward :: (Num a, Ord a) => NetParams a -> A.Array a -> A.Array a
forward = runPara model

linear1B ::
  (Num a, Fractional a) =>
  Para (NetParams a) (A.Array a) (Store (A.Array a) (A.Array a))
linear1B = Para $ \(p, x) ->
  Store
    (A.mult (A.transpose (w1 p)))
    (A.mult (w1 p) x)

bias1B ::
  (Num a) =>
  Para (NetParams a) (A.Array a) (Store (A.Array a) (A.Array a))
bias1B = Para $ \(p, x) ->
  Store id (x + b1 p)

relu1B ::
  (Ord a, Num a) =>
  Para (NetParams a) (A.Array a) (Store (A.Array a) (A.Array a))
relu1B = Para $ \(_, x) ->
  Store
    (A.zipWith (\xi dyi -> if xi > 0 then dyi else 0) x)
    (fmap (max 0) x)

andThen ::
  Para p a (Store b a) ->
  Para p b (Store c b) ->
  Para p a (Store c a)
andThen f g = Para $ \(p, a) ->
  case unPara f (p, a) of
    Store bwd_a b ->
      case unPara g (p, b) of
        Store bwd_b c ->
          Store (bwd_a . bwd_b) c

modelB ::
  (Ord a, Num a, Fractional a) =>
  Para (NetParams a) (A.Array a) (Store (A.Array a) (A.Array a))
modelB = linear1B `andThen` bias1B `andThen` relu1B

runModelB ::
  (Ord a, Num a, Fractional a) =>
  NetParams a -> A.Array a -> (A.Array a, A.Array a)
runModelB p x =
  case unPara modelB (p, x) of
    Store bwd y -> (y, bwd y)

data BackPass b a s p = BackPass
  { inputGrad :: b -> a,
    paramUpdate :: b -> s -> p -> p
  }

linear1BP ::
  (Num a, Fractional a) =>
  Para
    (NetParams a)
    (A.Array a)
    (Store (A.Array a) (BackPass (A.Array a) (A.Array a) a (NetParams a)))
linear1BP = Para $ \(p, x) ->
  Store
    ( \_ ->
        BackPass
          (A.mult (A.transpose (w1 p)))
          ( \dy' lr params ->
              params
                { w1 = w1 params - fmap (lr *) (A.expand (*) dy' x),
                  b1 = b1 params - fmap (lr *) dy'
                }
          )
    )
    (A.mult (w1 p) x)

-- | MSE loss: L = (1/n) sum (y - y')^2
mseLoss ::
  (Num a, Fractional a) =>
  A.Array a -> A.Array a -> (a, A.Array a)
mseLoss y target =
  let diff = y - target
      n = fromIntegral (A.size y)
      loss = sum (fmap (^ (2 :: Int)) diff) / n
      grad = fmap (* (2 / n)) diff
   in (loss, grad)

step ::
  (Ord a, Num a, Fractional a) =>
  a -> NetParams a -> A.Array a -> A.Array a -> (a, NetParams a)
step lr p x target =
  let y = forward p x
      (loss, dOut) = mseLoss y target
      Store mkBP _ = unPara linear1BP (p, x)
      bp = mkBP dOut
      p' = paramUpdate bp dOut lr p
   in (loss, p')

bias1BP ::
  (Num a) =>
  Para
    (NetParams a)
    (A.Array a)
    (Store (A.Array a) (BackPass (A.Array a) (A.Array a) a (NetParams a)))
bias1BP = Para $ \(p, x) ->
  Store
    ( \_ ->
        BackPass
          id
          ( \dy' lr params ->
              params
                { b1 = b1 params - fmap (lr *) dy'
                }
          )
    )
    (x + b1 p)

relu1BP ::
  (Ord a, Num a) =>
  Para
    (NetParams a)
    (A.Array a)
    (Store (A.Array a) (BackPass (A.Array a) (A.Array a) a (NetParams a)))
relu1BP = Para $ \(_, x) ->
  Store
    ( \_ ->
        BackPass
          (A.zipWith (\xi dyi -> if xi > 0 then dyi else 0) x)
          (\_ _ params -> params)
    )
    (fmap (max 0) x)

linear2BP ::
  (Num a, Fractional a) =>
  Para
    (NetParams a)
    (A.Array a)
    (Store (A.Array a) (BackPass (A.Array a) (A.Array a) a (NetParams a)))
linear2BP = Para $ \(p, x) ->
  Store
    ( \_ ->
        BackPass
          (A.mult (A.transpose (w2 p)))
          ( \dy' lr params ->
              params
                { w2 = w2 params - fmap (lr *) (A.expand (*) dy' x),
                  b2 = b2 params - fmap (lr *) dy'
                }
          )
    )
    (A.mult (w2 p) x)

bias2BP ::
  (Num a) =>
  Para
    (NetParams a)
    (A.Array a)
    (Store (A.Array a) (BackPass (A.Array a) (A.Array a) a (NetParams a)))
bias2BP = Para $ \(p, x) ->
  Store
    ( \_ ->
        BackPass
          id
          ( \dy' lr params ->
              params
                { b2 = b2 params - fmap (lr *) dy'
                }
          )
    )
    (x + b2 p)

andThenBP ::
  Para p a (Store b (BackPass b a s p)) ->
  Para p b (Store c (BackPass c b s p)) ->
  Para p a (Store c (BackPass c a s p))
andThenBP f g = Para $ \(p, a) ->
  case unPara f (p, a) of
    Store mkBP_a b ->
      case unPara g (p, b) of
        Store mkBP_b c ->
          Store
            ( \dc ->
                let bp_b = mkBP_b dc
                    bp_a = mkBP_a (inputGrad bp_b dc)
                 in BackPass
                      (inputGrad bp_a . inputGrad bp_b)
                      ( \dc' lr params ->
                          paramUpdate
                            bp_a
                            (inputGrad bp_b dc')
                            lr
                            (paramUpdate bp_b dc' lr params)
                      )
            )
            c

modelBP ::
  (Ord a, Num a, Fractional a) =>
  Para
    (NetParams a)
    (A.Array a)
    (Store (A.Array a) (BackPass (A.Array a) (A.Array a) a (NetParams a)))
modelBP =
  linear1BP
    `andThenBP` bias1BP
    `andThenBP` relu1BP
    `andThenBP` linear2BP
    `andThenBP` bias2BP

stepFull ::
  (Ord a, Num a, Fractional a) =>
  a -> NetParams a -> A.Array a -> A.Array a -> (a, NetParams a)
stepFull lr p x target =
  let Store mkBP y = unPara modelBP (p, x)
      (loss, dOut) = mseLoss y target
      bp = mkBP dOut
      p' = paramUpdate bp dOut lr p
   in (loss, p')
