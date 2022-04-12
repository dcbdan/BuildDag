module BuildDag.Kernel (
  getOutputDims, getIncidentDims, getAggRanks, getAggOp, getOrderings
) where

import Data.IntSet ( IntSet, (\\) )
import qualified Data.IntSet as IntSet

import Data.IntMap ( IntMap )
import qualified Data.IntMap as IntMap

import BuildDag.Types

import BuildDag.Misc ( idxInterval )

import Control.Monad.State ( State, execState, get, put )

getAggInfo :: Kernel ->  Maybe (CastableBOp, IntSet)

getAggInfo (KI_Contraction lhs rhs out _) |
  not (checkContraction lhs rhs out) = error "invalid args on KI_Contraction"
getAggInfo (KI_Contraction lhs rhs out _) =
  let nJoined = 1 + maximum (lhs ++ rhs)
      aggd    = joinedModes \\ outModes
      joinedModes = IntSet.fromList (idxInterval nJoined)
      outModes    = IntSet.fromList out
   in Just (CastableAdd, aggd)

getAggInfo (KI_Reduction op _ aggd _) = Just (op, aggd)
getAggInfo (KI_EW _ _ _)       = Nothing
getAggInfo (KI_EWB _ _ _ _ _ ) = Nothing
getAggInfo (KI_Dropout _ _)    = Nothing

-- Each kernel has an incident rank R and they are labeled
--   [0,...,R-1].
-- This function gets for each input the input labels
getOrderings :: Kernel -> [[Rank]]
getOrderings (KI_Contraction lhs rhs _ _) = [lhs, rhs]
getOrderings (KI_Reduction _ n _ _) = [idxInterval n]
getOrderings (KI_EW _ ret _) = [ret]
getOrderings (KI_EWB _ lhs rhs _ _) = [lhs, rhs]
getOrderings (KI_Dropout n _) = [idxInterval n]

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

getAggRanks :: Kernel -> IntSet
getAggRanks kernel =
  case getAggInfo kernel of
    Nothing       -> IntSet.empty
    Just (_, ret) -> ret

getAggOp :: Kernel -> CastableBOp
getAggOp kernel =
  case getAggInfo kernel of
    Nothing       -> error "This op has no associated agg"
    Just (ret, _) -> ret

