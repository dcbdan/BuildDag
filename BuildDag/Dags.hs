module BuildDag.Dags(
  parseDagArgs, dagUsage
) where

import BuildDag.Dags.ChainMatMul
import BuildDag.Dags.AmazonCat13K
import qualified BuildDag.Dags.Bert as Bert
import qualified BuildDag.Dags.BHT as BHT
import qualified BuildDag.Dags.Ff as Ff
import qualified BuildDag.Dags.Exp as Exp

--------------------------------------------------------------------

import BuildDag.Types
import qualified BuildDag.Graph as Graph

import qualified Control.Monad.RWS.Lazy as RWS

import Text.Read ( readMaybe )

import Data.Map ( Map )
import qualified Data.Map as Map

dagUsage :: [String]
dagUsage = [
    "amazonCat13K batchSize hiddenSize",
    "amazonCat13K batchSize hiddenSize nIter",
    "bert batchSize",
    "bert batchSize numLayers nQuery nHead nSequence",
    "bert batchSize numLayers nQuery nHead nSequence dropout",
    "matmul nI nJ nK                   [for ij,jk->ik]",
    "7matmul n",
    "ff nB nInn nOut nHidden",
    "bertPerturb batchSize numLayers nQuery nHead nSequence",
    "bht batchSize nSequence nHidden nHead nHH",
    "exp which"
  ]

parseDagArgs :: String -> [String] -> Maybe Dag

parseDagArgs "amazonCat13K" (nStr:hStr:[]) = do
  let p = 203882
      l = 13330
  n <- readDimension nStr
  h <- readDimension hStr
  return $ (uncurry getDag) (amazonCat13K n h 1)

parseDagArgs "amazonCat13K" (nStr:hStr:nIterStr:[]) = do
  let p = 203882
      l = 13330
  n <- readDimension nStr
  h <- readDimension hStr
  nIter <- readDimension nIterStr
  return $ (uncurry getDag) (amazonCat13K n h nIter)


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

parseDagArgs "7matmul" [nStr] = do
  n <- readDimension nStr
  let name n = "X" ++ show n
      f n1 n2 = CMNode (CMLeaf (name n1)) (CMLeaf (name n2))
      l1 = f 1 2
      l2 = f 3 4
      l3 = f 5 6
      l4 = f 7 8
      m1 = CMNode l1 l2
      m2 = CMNode l3 l4
      dagM = chainMatMul $ CMNode m1 m2
      sizes = Map.fromList $ zip (map name [1..8]) (replicate 8 [n,n])
  return $ getDag sizes dagM

parseDagArgs "ff" s | length s == 4 = do
  ds <- mapM readDimension s
  let [nB,nInn,nOut,nHidden] = ds
  return $ (uncurry getDag) (Ff.ff nB nInn nOut nHidden)

parseDagArgs "bertPerturb" (nBatchStr:nLayerStr:nQueryStr:nHeadStr:nSequenceStr:[]) = do
  nBatch    <- readDimension nBatchStr
  nLayer    <- readDimension nLayerStr
  nQuery    <- readDimension nQueryStr
  nHead     <- readDimension nHeadStr
  nSequence <- readDimension nSequenceStr

  let params = Bert.Params {
          Bert._nLayer    = nLayer,
          Bert._nQuery    = nQuery,
          Bert._nHead     = nHead,
          Bert._nSequence = nSequence,
          Bert._fDropout  = (-1.0) -- no dropout
        }

  return $ (uncurry getDag) (Bert.bertPerturb nBatch params)

parseDagArgs "bht" (nBatchStr:nSeqStr:nHiddenStr:nHeadStr:nHHStr:[]) = do
  nBatch  <- readDimension nBatchStr
  nSeq    <- readDimension nSeqStr
  nHidden <- readDimension nHiddenStr
  nHead   <- readDimension nHeadStr
  nHH     <- readDimension nHHStr

  let params = BHT.Params {
          BHT._nB  = nBatch,
          BHT._nS  = nSeq,
          BHT._nH  = nHidden,
          BHT._nN  = nHead,
          BHT._nHH = nHH
        }

  return $ (uncurry getDag) (BHT.bht params)

parseDagArgs "exp" (which:[]) = do
  w <- readInt which
  case w of
    1 -> return $ (uncurry getDag) (Exp.exp01)
    _ -> Nothing

parseDagArgs _ _ = Nothing

getDag :: Map String Dims -> (BuildDagM a) -> Dag
getDag lookupTable doThis = fst $ RWS.execRWS doThis lookupTable Graph.empty

readDimension :: String -> Maybe Dim
readDimension = readMaybe

readInt = readDimension

readFloat :: String -> Maybe Float
readFloat = readMaybe
