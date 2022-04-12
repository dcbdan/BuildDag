module BuildDag.Dags(
  module BuildDag.Dags.ChainMatMul,
  parseDagArgs
) where

import BuildDag.Dags.ChainMatMul
import BuildDag.Dags.AmazonCat13K

--------------------------------------------------------------------

import BuildDag.Types
import qualified BuildDag.Graph as Graph

import qualified Control.Monad.RWS.Lazy as RWS

import Text.Read ( readMaybe )

import Data.Map ( Map )
import qualified Data.Map as Map

parseDagArgs :: String -> [String] -> Maybe Dag
parseDagArgs "amazonCat13K" (nStr:hStr:[]) = do
  let p = 203882
      l = 13330
  n <- readDimension nStr
  h <- readDimension hStr
  return $ (uncurry getDag) (amazonCat13K n h)
parseDagArgs "matmul" s | length s == 3 = do
  ds <- mapM readDimension s
  let [i,j,k] = ds
  let dagM = chainMatMul $ CMNode (CMLeaf "X") (CMLeaf "Y")
      sizes = Map.fromList [("X", [i,j]), ("Y", [j,k])]
  return $ getDag sizes dagM
parseDagArgs _ _ = Nothing

getDag :: Map String Dims -> (BuildDagM a) -> Dag
getDag lookupTable doThis = fst $ RWS.execRWS doThis lookupTable Graph.empty

readDimension :: String -> Maybe Dim
readDimension = readMaybe

