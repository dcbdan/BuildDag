module BuildDag.Dags(
  module BuildDag.Dags.ChainMatMul,
  parseDagArgs
) where

import BuildDag.Dags.ChainMatMul
import BuildDag.Dags.AmazonCat13K
import qualified BuildDag.Dags.Bert as Bert

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

-- TODO: what are the default sizes?
parseDagArgs "bert" sizes@(nBatchStr:[]) =
  parseDagArgs "bert"     (nBatchStr:"2"      :"8192"   :"4"     :"4096"      :[])

parseDagArgs "bert" sizes@(nBatchStr:nLayerStr:nQueryStr:nHeadStr:nSequenceStr:[]) =
  parseDagArgs "bert" (sizes ++ ["-1.0"])
parseDagArgs "bert" (nBatchStr:nLayerStr:nQueryStr:nHeadStr:nSequenceStr:fDropoutStr:[]) = do
  nBatch    <- readDimension nBatchStr
  nLayer    <- readDimension nLayerStr
  nQuery    <- readDimension nQueryStr
  nHead     <- readDimension nHeadStr
  nSequence <- readDimension nSequenceStr

  fDropout <- readFloat fDropoutStr
  let params = Bert.Params {
          Bert._nLayer    = nLayer,
          Bert._nQuery    = nQuery,
          Bert._nHead     = nHead,
          Bert._nSequence = nSequence,
          Bert._fDropout  = fDropout
        }

  return $ (uncurry getDag) (Bert.bert nBatch params)

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

readFloat :: String -> Maybe Float
readFloat = readMaybe
