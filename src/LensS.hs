{-# LANGUAGE TupleSections #-}

module LensS
  ( LensS (..),
    Store (..),
    mkLensS,
    getS,
    setS,
  )
where

import Control.Arrow (Arrow (..))
import Control.Category (Category (..))
import Data.Profunctor (Profunctor (..), Strong (..))
import Para (Para (..))
import Prelude hiding (id, (.))
import Prelude qualified

-- | @Store b a = (b -> a, b)@
data Store b a = Store (b -> a) b

instance Functor (Store b) where
  fmap f (Store g b) = Store (f Prelude.. g) b

pos :: Store b a -> b
pos (Store _ b) = b

peek :: Store b a -> b -> a
peek (Store f _) b = f b

-- | @LensS a b = a -> Store b a@
newtype LensS a b = LensS {runLensS :: a -> Store b a}

instance Category LensS where
  id = LensS $ \a -> Store Prelude.id a

  LensS f . LensS g = LensS $ \a ->
    case g a of
      Store sba b ->
        case f b of
          Store scb c ->
            Store (sba Prelude.. scb) c

-- | Build from get and set.
mkLensS :: (a -> b) -> (a -> b -> a) -> LensS a b
mkLensS get set = LensS $ \a -> Store (set a) (get a)

-- | Get the focus.
getS :: LensS a b -> a -> b
getS (LensS f) a = pos (f a)

-- | Set the focus.
setS :: LensS a b -> a -> b -> a
setS (LensS f) a b = peek (f a) b
