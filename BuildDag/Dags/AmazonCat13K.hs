module BuildDag.Dags.AmazonCat13K ( amazonCat13K ) where

import Prelude hiding ( subtract )

import Data.Map ( Map )
import qualified Data.Map as Map

import Control.Monad ( foldM )

import BuildDag.Types
import BuildDag.Build

amazonCat13K :: Int -> Int -> Int -> (Map String Dims, BuildDagM ())
amazonCat13K n h nIter = (inputs n p h l, build nIter)

p = 203882
l = 13330

inputs n p h l = Map.fromList [
  ("X", [p,n]),
  ("Y", [l,n]),
  ("W1", [h,p]),
  ("W2", [l,h]) ]

learningRate :: Float
learningRate = 0.01

iter x y (w1, w2) = do
  s1 <- matmul w1 x               -- hn
  a1 <- elementwise Relu [0,1] s1 -- hn

  s2 <- matmul w2 a1                 -- ln
  a2 <- elementwise Sigmoid [0,1] s2 -- ln

  ga2 <- subtract a2 y -- ln
  gw2 <- matmul_TAlpha learningRate ga2 a1 -- lh

  tga1 <- matmulT_ w2 ga2                -- hn
  tra1 <- elementwise Reluderiv [0,1] a1 -- hn
  ga1  <- hadamard tga1 tra1             -- hn

  gw1 <- matmul_TAlpha learningRate ga1 x -- hp

  w1_ <- subtract w1 gw1
  w2_ <- subtract w2 gw2

  return (w1_, w2_)

build :: Int -> BuildDagM ()
build nIter = do
  -- w: weight
  -- g: grad
  -- t: tmp
  -- s: tmp
  x  <- initFile "X" 1 -- pn
  y  <- initFile "Y" 2 -- ln
  w1 <- initRandom "W1" (-0.1) 0.1 -- hp
  w2 <- initRandom "W2" (-0.1) 0.1 -- lh

  let f params whichIter = iter x y params
  (w1Out, w2Out) <- foldM f (w1,w2) [1..nIter]

  return ()

matmul_TAlpha alpha = contractionAlpha [0,1] [2,1] [0,2] alpha
matmulT_Alpha alpha = contractionAlpha [1,0] [1,2] [0,2] alpha
