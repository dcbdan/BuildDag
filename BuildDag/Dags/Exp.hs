module BuildDag.Dags.Exp (
  exp01, exp02
) where

import Data.Map ( Map )
import qualified Data.Map as Map

import BuildDag.Types
import BuildDag.Build

exp01 :: (Map String Dims, BuildDagM ())
exp01 = (inputs01, build01)

inputs01 :: Map String Dims
inputs01 = Map.fromList [
  ("x", [j, i]),
  ("y", [k])]
  where i = 4
        j = 6
        k = i*j

build01 :: BuildDagM ()
build01 = do
  -- Note: (1) It is illegal in from dag to (a) have multiple reblocks in a row,
  --           and to (b) end with a reblock.
  --       (2) Mergesplits are sneaky reblocks
  --       =>  It is illegal in from dag to (a) have multiple mergesplits in a row,
  --           and to (b) end with a reblock.
  let uop = AddScalar 0.0

  x   <- initRandom "x" (-1.0) (1.0)
  xx  <- merge x
  xz  <- elementwise uop [0] xx
  xxx <- split 6 xz
  xxz <- elementwise uop [0,1] xxx

  y   <- initRandom "y" (-1.0) (1.0)
  yy  <- split 6 y
  yz  <- elementwise uop [0,1] yy
  yyy <- merge yz
  yyz <- elementwise uop [0] yyy

  return ()

exp02 :: (Map String Dims, BuildDagM ())
exp02 = (inputs02, build02)

inputs02 :: Map String Dims
inputs02 = Map.fromList [
  ("x", [l, i, j])]
  where i = 4
        j = 6
        k = i*j
        l = 1024

build02 :: BuildDagM ()
build02 = do
  x_lij <- initRandom "x" (-1.0) (1.0)
  x_jil <- transpose 0 2 x_lij
  x_kl  <- merge x_jil
  ret   <- elementwise (AddScalar 123.321) [0,1] x_kl
  return ()
