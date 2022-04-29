module BuildDag.Dags.Ff(
  ff
) where

import Prelude hiding ( subtract )

import Data.Map ( Map )
import qualified Data.Map as Map

import BuildDag.Types
import BuildDag.Build

ff :: Int -> Int -> Int -> Int -> (Map String Dims, BuildDagM ())
ff nB nI nO nH = (inputs nB nI nO nH, build)

inputs nB nI nO nH = Map.fromList [
  ("X", [nI,nB]),
  ("Y", [nO,nB]),
  ("W1", [nH,nI]),
  ("W2", [nO,nH]) ]

learningRate :: Float
learningRate = 0.01

build :: BuildDagM ()
build = do
  -- w: weight
  -- g: grad
  -- t: tmp
  -- s: tmp
  x  <- initRandom "X"  (-1.0) 1.0 -- ib
  y  <- initRandom "Y"  (-1.0) 1.0 -- ob
  w1 <- initRandom "W1" (-1.0) 1.0 -- hi
  w2 <- initRandom "W2" (-1.0) 1.0 -- oh

  s1 <- matmul w1 x
  a1 <- elementwise Relu [0,1] s1

  s2 <- matmul w2 a1
  a2 <- elementwise Sigmoid [0,1] s2

  ga2 <- subtract a2 y
  gw2 <- matmul_TAlpha learningRate ga2 a1

  tga1 <- matmulT_ w2 ga2
  tra1 <- elementwise Reluderiv [0,1] a1
  ga1  <- hadamard tga1 tra1

  gw1 <- matmul_TAlpha learningRate ga1 x

  w1_ <- subtract w1 gw1
  w2_ <- subtract w2 gw2

  return ()

matmul_TAlpha alpha = contractionAlpha [0,1] [2,1] [0,2] alpha
matmulT_Alpha alpha = contractionAlpha [1,0] [1,2] [0,2] alpha
