module BuildDag.Dags.BHT ( -- BHT == BinHang's Transformer
  Params(..), bht
) where

import Data.Map ( Map )
import qualified Data.Map as Map

import BuildDag.Types
import BuildDag.Module
import BuildDag.Build

data Params = Params {
    _nB  :: Int,
    _nS  :: Int,
    _nH  :: Int,
    _nN  :: Int,
    _nHH :: Int
  }

bht :: Params -> (Map String Dims, BuildDagM ())
bht = undefined

