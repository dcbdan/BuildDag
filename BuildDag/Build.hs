module BuildDag.Build (
  matmul, matmul__, matmulT_, matmul_T,
  contraction, contractionAlpha,
  reduction, reductionAlpha,
  elementwise, elementwiseAlpha,
  elementwiseBinary, elementwiseBinaryAlpha,
  dropout, add, subtract, hadamard,
  initRandom, initConstant,
  ----------------
  liftGraph, getObject
) where

import Prelude hiding ( subtract )

import Data.IntSet ( IntSet )
import qualified Data.IntSet as IntSet

import qualified Data.Map as Map

import qualified Control.Monad.RWS.Lazy as RWS
import Control.Monad.Reader ( ask )

import BuildDag.Graph ( Graph, GraphM )
import qualified BuildDag.Graph as Graph

import BuildDag.Types

import qualified BuildDag.Kernel as K

import BuildDag.Misc ( idxInterval )

matmul :: Id -> Id -> BuildDagM Id

matmul   = contraction [0,1] [1,2] [0,2]
matmul__ = matmul
matmulT_ = contraction [1,0] [1,2] [0,2]
matmul_T = contraction [0,1] [2,1] [0,2]

contraction :: [Rank] -> [Rank] -> [Rank] -> Id -> Id -> BuildDagM Id
contraction a b c d e = contractionAlpha a b c 1.0 d e

contractionAlpha :: [Rank] -> [Rank] -> [Rank] -> Float -> Id -> Id -> BuildDagM Id
contractionAlpha lhsModes rhsModes outModes alpha lhs rhs |
  not (checkContraction lhsModes rhsModes outModes) =
    error $ "contraction error" ++
      (show lhsModes) ++ ", " ++ (show rhsModes) ++ ", " ++ (show outModes)
contractionAlpha lhsModes rhsModes outModes alpha lhs rhs =
  let joinKernel = KI_Contraction lhsModes rhsModes outModes alpha
   in do lhsReblock <- _reblock lhs
         rhsReblock <- _reblock rhs
         outJoin <- _join joinKernel [lhsReblock,rhsReblock]
         if IntSet.null (K.getAggRanks joinKernel)
            then return outJoin
            else _agg (K.getAggOp joinKernel) outJoin

reduction         a b d   = reductionAlpha a b 1.0 d

reductionAlpha :: CastableBOp -> IntSet -> Float -> Id -> BuildDagM Id
reductionAlpha op aggd alpha inn =
  let getReductionKernel rankIn = KI_Reduction op rankIn aggd alpha
   in do rankIn <- getOutputRank inn
         let reductionKernel = getReductionKernel rankIn
         innReblock <- _reblock inn
         outJoin <- _join reductionKernel [innReblock]
         if IntSet.null (K.getAggRanks reductionKernel)
            then error "Why is this reduction kernel a no op?"
            else _agg (K.getAggOp reductionKernel) outJoin


elementwise a b d = elementwiseAlpha a b 1.0 d

-- This op does not do an input reblocking or a post aggregation.
elementwiseAlpha :: UOp -> [Rank] -> Float -> Id -> BuildDagM Id
elementwiseAlpha op outModes alpha inn =
  let joinKernel = KI_EW op outModes alpha
   in do rankIn <- getOutputRank inn
         if rankIn /= length outModes
            then error "rankIn is incrrect size in EW"
            else _join joinKernel [inn]

elementwiseBinary a b c f g = elementwiseBinaryAlpha a b c 1.0 1.0 f g

-- This op does not do an input reblocking or a post aggregation.
elementwiseBinaryAlpha ::
  BOp ->
  [Int] -> [Int] ->
  Float -> Float ->
  Id -> Id ->
  BuildDagM Id
elementwiseBinaryAlpha op lhsModes rhsModes alpha beta lhs rhs =
  let kernel = KI_EWB op lhsModes rhsModes alpha beta
   in _join kernel [lhs, rhs]

dropout :: Float -> Id -> BuildDagM Id
dropout f inn = do
  n <- getOutputRank inn
  _join (KI_Dropout n f) [inn]

_same_dim_binary :: BOp -> Id -> Id -> BuildDagM Id
_same_dim_binary op lhs rhs = do
  lhsDims <- getOutputDims lhs
  rhsDims <- getOutputDims rhs
  let errMsg = "this must be applied to items with the same dimensions.. "++
               "try elementwise binary"
      interval = idxInterval (length lhsDims)
  if (lhsDims /= rhsDims)
     then error errMsg
     else elementwiseBinary op interval interval lhs rhs

add      = _same_dim_binary Add
hadamard = _same_dim_binary Mul
subtract = _same_dim_binary Sub

initRandom :: String -> Float -> Float -> BuildDagM Id
initRandom name a b = _input name (InitRandom (min a b) (max a b))

initConstant :: String -> Float -> BuildDagM Id
initConstant name val = _input name (InitConstant val)

--------------------------------------------------------------------------------

_input :: String -> Init -> BuildDagM Id
_input name init = do
  lookupTable <- ask
  dims <- case name `Map.lookup` lookupTable of
            Nothing -> error (name ++ "'s dimensions are not in the lookup table")
            Just dims -> return dims
  liftGraph $ do
    let newInputNode id = Node id [] dims (Input init)
    Graph.insertObjectWithId newInputNode

_reblock :: Id -> BuildDagM Id
_reblock inn = do
  innDims <- getOutputDims inn
  let newReblockNode out = Node out [inn] innDims Reblock
  liftGraph $ Graph.insertObjectWithId newReblockNode

_agg :: CastableBOp -> Id -> BuildDagM Id
_agg op inn = do
  dims <- getOutputDims inn
  let newAggNode out = Node out [inn] dims (Agg op)
  liftGraph $ Graph.insertObjectWithId newAggNode

_join :: Kernel -> [Id] -> BuildDagM Id
_join kernel inns = do
  inputDims <- mapM getOutputDims inns
  let incDims = K.getIncidentDims kernel inputDims
      newJoinNode out = Node out inns incDims (Join kernel)
  liftGraph $ Graph.insertObjectWithId newJoinNode

liftGraph :: DagM a -> BuildDagM a
liftGraph doThis =
  let f _ innState =
        let (result, outState) = Graph.runGraphM doThis innState
         in (result, outState, ())
   in RWS.rws f

getObject :: Id -> BuildDagM Node
getObject id = liftGraph (Graph.getObject id)

getIncidentRank :: Id -> BuildDagM Rank
getIncidentRank id = (length . _incDims) <$> getObject id

getAggRanks :: Id -> BuildDagM IntSet
getAggRanks id = do
  info <- _info <$> getObject id
  return $ case info of
    Input _ -> IntSet.empty
    Join k  -> K.getAggRanks k
    Reblock -> IntSet.empty
    Agg _   -> IntSet.empty

getOutputRank :: Id -> BuildDagM Rank
getOutputRank id = do
  nInc <- getIncidentRank id
  nAgg <- IntSet.size <$> getAggRanks id
  return (nInc - nAgg)

getIncidentDims :: Id -> BuildDagM Dims
getIncidentDims id = _incDims <$> getObject id

getOutputDims :: Id -> BuildDagM Dims
getOutputDims id = do
  incDims <- getIncidentDims id
  aggd <- getAggRanks id
  return $ incToOut aggd incDims

