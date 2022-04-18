module BuildDag.Kernel (
  getOutputDims, getIncidentDims, getAggRanks, getAggOp, getOrderings
) where

import Data.IntSet ( IntSet, (\\), union )
import qualified Data.IntSet as IntSet

import Data.IntMap ( IntMap )
import qualified Data.IntMap as IntMap

import BuildDag.Types

import BuildDag.Misc ( idxInterval )

import Control.Monad.State ( State, execState, get, put )

getIncOutRank :: Kernel -> (Int, Int)
getIncOutRank (KI_Contraction lhs rhs out _) |
  not (checkContraction lhs rhs out) = error "invalid args on KI_Contraction"
getIncOutRank (KI_Contraction lhs rhs out _) =
  let nInc = IntSet.size (lhsS `union` rhsS)
      nOut = length out
      lhsS = IntSet.fromList lhs
      rhsS = IntSet.fromList rhs
   in (nInc, nOut)
getIncOutRank (KI_Reduction _ nInc out _) = (nInc, length out)
getIncOutRank (KI_EW _ out _) = (n, n)
  where n = length out
getIncOutRank (KI_EWB _ lhs rhs out _) =
  let nInc = IntSet.size (lhsS `union` rhsS)
      nOut = length out
      lhsS = IntSet.fromList lhs
      rhsS = IntSet.fromList rhs
   in if nInc /= nOut
         then error "Invalid EWB"
         else (nInc, nOut)

-- Each kernel has an incident rank R and they are labeled
--   [0,...,R-1].
-- This function gets for each input the input labels
getOrderings :: Kernel -> [[Rank]]

getOrderings k@(KI_Contraction lhs rhs out _) | checkContraction lhs rhs out =
  let aggd = IntSet.toList $ (lhsS `union` rhsS) \\ outS
      lhsS = IntSet.fromList lhs
      rhsS = IntSet.fromList rhs
      outS = IntSet.fromList out
      outModified = out ++ aggd
   in getInputOrderings [lhs,rhs] outModified
getOrderings (KI_Contraction _ _ _ _) = error "invalid KI_Contraction"

getOrderings k@(KI_Reduction _ n outModes _) =
  let aggd = IntSet.toList $ innS \\ outS
      innS = IntSet.fromList $ idxInterval n
      outS = IntSet.fromList outModes
      outModified = outModes ++ aggd
   in getInputOrderings [idxInterval n] outModified

getOrderings (KI_EW _ outModes _) = getInputOrderings [idxInterval n] outModes
  where n = length outModes

getOrderings (KI_EWB _ lhs rhs out _) = getInputOrderings [lhs, rhs] out

-- Get the orderings of the inputs and use that to deduce
-- the incident dimensions. If any of the deductions do not
-- make sense, violently die.
getIncidentDims :: Kernel -> [Dims] -> Dims
getIncidentDims kernel innDims =
  let orderings = getOrderings kernel

      -- because the 400 different types of folds gets confusing,
      -- here, use a state monad.
      addDim :: (Rank, Dim) -> State (IntMap Dim) ()
      addDim (rank, dim) = do
        m <- get
        case rank `IntMap.lookup` m of
          Nothing -> put $ IntMap.insert rank dim m
          Just otherDim | otherDim == dim -> return ()
          Just otherDim                   -> error "Incorrect dimensions in matching"

      addOrdering (ordering, dims) = mapM addDim (zip ordering dims)

      addAll = mapM addOrdering (zip orderings innDims)

      retMap = execState addAll IntMap.empty
   in if (IntMap.keys retMap /= idxInterval (IntMap.size retMap))
         then error "these keys are not correct"
         else map snd (IntMap.toAscList retMap)

getOutputDims :: Kernel -> [Dims] -> Dims
getOutputDims k ds =
  let inc = getIncidentDims k ds
      aggd = getAggRanks k
   in incToOut aggd inc

-- The convention for KI_Contraction and KI_Reduction is to tag
-- the agg modes on at the end. So
--  ijk -> ik reduces to
--  ikj    ij
getAggRanks :: Kernel -> IntSet
getAggRanks kernel =
  let (nInc, nOut) = getIncOutRank kernel
   in IntSet.fromList [nOut..(nInc-1)]

getAggOp :: Kernel -> CastableBOp
getAggOp (KI_Contraction _ _ _ _) = CastableAdd
getAggOp (KI_Reduction op _ _ _) = op
getAggOp _ = error "This op has no associated agg"

