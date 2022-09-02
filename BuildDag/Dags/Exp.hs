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
  x   <- initRandom "x" (-1.0) (1.0)
  xx  <- merge x
  xxx <- split 6 xx

  y   <- initRandom "y" (-1.0) (1.0)
  yy  <- split 6 y
  yyy <- merge yy

  return ()

exp02 :: (Map String Dims, BuildDagM ())
exp02 = undefined
