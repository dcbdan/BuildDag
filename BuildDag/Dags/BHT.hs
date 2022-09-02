module BuildDag.Dags.BHT ( -- BHT == BinHang's Transformer
  Params(..), bht
) where

import Data.Map ( Map )
import qualified Data.Map as Map

import BuildDag.Types
import BuildDag.Build

data Params = Params {
    _nB  :: Int,
    _nS  :: Int,
    _nH  :: Int,
    _nN  :: Int,
    _nHH :: Int
  }

bht :: Params -> (Map String Dims, BuildDagM ())
bht params = (inputs params, build)

inputs (Params nB nS nH nN nHH) = Map.fromList [
  ("x", [nH,nS,nB]),
  ("wq", [nH,nH]),
  ("wv", [nH,nH]),
  ("wk", [nH,nH]),
  ("wo", [nH,nH]),
  ("w1", [4*nH, nH]),
  ("w2", [nH, 4*nH])]

build :: BuildDagM ()
build = do
  x  <- initRandom "x"  (-1.0) 1.0
  wq <- initRandom "wq" (-1.0) 1.0
  wv <- initRandom "wv" (-1.0) 1.0
  wk <- initRandom "wk" (-1.0) 1.0
  wo <- initRandom "wo" (-1.0) 1.0
  w1 <- initRandom "w1" (-1.0) 1.0
  w2 <- initRandom "w2" (-1.0) 1.0

  xq <- mm x wq -- [h1,s,b],[h2,h1]->[h2,s,b]
  xqp <- fsplit xq
  xqpp <- ftranspose [1,2] xqp

  return ()

mm = contractionAlpha [2,1,0] [3,2] [3,1,0] 1.0

fsplit = undefined
ftranspose = undefined
fmerge = undefined
