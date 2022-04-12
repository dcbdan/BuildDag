module BuildDag.Dags.AmazonCat13K ( amazonCat13K ) where

import Prelude hiding ( subtract )

import Data.Map ( Map )
import qualified Data.Map as Map

import BuildDag.Types
import BuildDag.Build

amazonCat13K :: Int -> Int -> (Map String Dims, BuildDagM ())
amazonCat13K n h = (inputs n p h l, build)

p = 203882
l = 13330

inputs n p h l = Map.fromList [
  ("X", [p,n]),
  ("Y", [l,n]),
  ("W1", [h,p]),
  ("W2", [l,h]) ]

build :: BuildDagM ()
build = do
  -- w: weight
  -- g: grad
  -- t: tmp
  -- s: tmp
  x  <- initRandom "X"  (-1.0) 1.0 -- pn
  y  <- initRandom "Y"  (-1.0) 1.0 -- ln
  w1 <- initRandom "W1" (-1.0) 1.0 -- hp
  w2 <- initRandom "W2" (-1.0) 1.0 -- lh

  s1 <- matmul w1 x               -- hn
  a1 <- elementwise Relu [0,1] s1 -- hn

  s2 <- matmul w2 a1                 -- ln
  a2 <- elementwise Sigmoid [0,1] s2 -- ln

  ga2 <- subtract a2 y -- ln
  gw2 <- matmul_T ga2 a1 -- lh

  tga1 <- matmulT_ w2 ga2                -- hn
  tra1 <- elementwise Reluderiv [0,1] a1 -- hn
  ga1  <- hadamard tga1 tra1             -- hn

  gw1 <- matmul_T ga1 x -- hp

  return ()


