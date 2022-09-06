module BuildDag.Build (
  matmul, matmul__, matmulT_, matmul_T,
  contraction, contractionAlpha,
  reduction, reductionAlpha,
  elementwise, elementwiseAlpha,
  elementwiseBinary, elementwiseBinaryAlpha,
  dropout, add, subtract, hadamard, scale,
  initInput, initRandom, initConstant, initFile,
  merge, split, transpose, permute, einsum, einsumAlpha,
  ----------------
  liftGraph, getObject, getOutputDims
) where

import Prelude hiding ( subtract )

import Data.IntSet ( IntSet )
import qualified Data.IntSet as IntSet

import Data.Map ( Map )
import qualified Data.Map as Map

import qualified Control.Monad.RWS.Lazy as RWS
import Control.Monad.Reader ( ask )
import Control.Monad.State ( State, evalState, get, put )

import BuildDag.Graph ( Graph, GraphM )
import qualified BuildDag.Graph as Graph

import BuildDag.Types

import qualified BuildDag.Kernel as K

import BuildDag.Misc ( idxInterval, (.>) )
import qualified BuildDag.Misc as Misc ( split )

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
            then error "This is not a contraction, this is an EWB with +."
            else _agg (K.getAggOp joinKernel) outJoin

reduction         a b d   = reductionAlpha a b 1.0 d

reductionAlpha :: CastableBOp -> [Int] -> Float -> Id -> BuildDagM Id
reductionAlpha op outModes alpha inn =
  let getReductionKernel rankIn = KI_Reduction op rankIn outModes alpha
   in do rankIn <- getOutputRank inn
         let reductionKernel = getReductionKernel rankIn
         innReblock <- _reblock inn
         outJoin <- _join reductionKernel [innReblock]
         if IntSet.null (K.getAggRanks reductionKernel)
            then error "Why is this reduction kernel a no op?"
            else _agg (K.getAggOp reductionKernel) outJoin


elementwise a b d = elementwiseAlpha a b 1.0 d

-- Always do a reblock because _reblock makes sure the input isn't
-- a rogue merge. Downstream applications can do what they want with
-- the reblocks and merges--i.e. they should deal with ew-reblock-ew.
elementwiseAlpha :: UOp -> [Rank] -> Float -> Id -> BuildDagM Id
elementwiseAlpha op outModes alpha inn =
  let joinKernel = KI_EW op outModes alpha
   in do innReblock <- _reblock inn
         rankIn <- getOutputRank innReblock
         if rankIn /= length outModes
            then error "rankIn is incrrect size in EW"
            else _join joinKernel [innReblock]

elementwiseBinary a b c d g h = elementwiseBinaryAlpha a b c d 1.0 g h

-- There is no aggregation.
elementwiseBinaryAlpha ::
  BOp ->
  [Int] -> [Int] -> [Int] ->
  Float ->
  Id -> Id ->
  BuildDagM Id
elementwiseBinaryAlpha op lhsModes rhsModes outModes alpha lhs rhs =
  let kernel = KI_EWB op lhsModes rhsModes outModes alpha
   in do lhsReblock <- _reblock lhs
         rhsReblock <- _reblock rhs
         _join kernel [lhsReblock, rhsReblock]

dropout :: Float -> Id -> BuildDagM Id
dropout f inn = do
  n <- getOutputRank inn
  elementwise (Dropout f) (idxInterval n) inn

_same_dim_binary :: BOp -> Id -> Id -> BuildDagM Id
_same_dim_binary op lhs rhs = do
  lhsDims <- getOutputDims lhs
  rhsDims <- getOutputDims rhs
  let errMsg = "this must be applied to items with the same dimensions.. "++
               "try elementwise binary"
      interval = idxInterval (length lhsDims)
  if (lhsDims /= rhsDims)
     then error errMsg
     else elementwiseBinary op interval interval interval lhs rhs

add      = _same_dim_binary Add
hadamard = _same_dim_binary Mul
subtract = _same_dim_binary Sub

scale :: Float -> Id -> BuildDagM Id
scale f inn = do
  r <- getOutputRank inn
  let ds = [0..(r-1)]
  elementwise (MulScalar f) ds inn

initInput :: String -> Init -> BuildDagM Id
initInput = _input

initRandom :: String -> Float -> Float -> BuildDagM Id
initRandom name a b = _input name (InitRandom (min a b) (max a b))

initConstant :: String -> Float -> BuildDagM Id
initConstant name val = _input name (InitConstant val)

initFile :: String -> Int -> BuildDagM Id
initFile name whichFile = _input name (InitFile whichFile)

-- merge gives: ij->k where |k| = |i| * |j|.
merge :: Id -> BuildDagM Id
merge inn = do
  assertNotMergeOrReblock inn
  innDims <- getOutputDims inn
  let newMergeNode out = Node out [inn] outDims (MergeSplit Nothing)
      outDims = case innDims of
                  (i:j:rest) -> (i*j:rest)
  liftGraph $ Graph.insertObjectWithId newMergeNode

-- split |i| gives: k -> ij where |j| = |k|/|i|.
split :: Dim -> Id -> BuildDagM Id
split i inn = do
  assertNotMergeOrReblock inn
  innDims <- getOutputDims inn
  let newSplitNode out = Node out [inn] outDims (MergeSplit (Just i))
      outDims = case innDims of
                  (k:rest) | k `mod` i == 0 -> (i:(k `div` i):rest)
  liftGraph $ Graph.insertObjectWithId newSplitNode

transpose :: Rank -> Rank -> Id -> BuildDagM Id
transpose r0 r1 inn = do
  rankInn <- getOutputRank inn
  let trModes = map fix [0..(rankInn-1)]
      fix r | r == r0 = r1
      fix r | r == r1 = r0
      fix r           = r
  if r0 >= rankInn || r1 >= rankInn || r0 == r1
     then error "invalid transpose"
     else elementwise NoOp trModes inn

permute :: [Rank] -> Id -> BuildDagM Id
permute outModes inn = do
  rankInn <- getOutputRank inn
  if length outModes /= rankInn || outModes == [0..(rankInn-1)]
     then error "invalid permute"
     else elementwise NoOp outModes inn

einsum :: String -> Id -> Id -> BuildDagM Id
einsum einsumStr = einsumAlpha einsumStr 1.0

einsumAlpha :: String -> Float -> Id -> Id -> BuildDagM Id
einsumAlpha einsumStr f = contractionAlpha lhs rhs out f
  where (lhs, rhs, out) = fromEinsumString einsumStr

-- Einsum string, loosefly of the form
--   "batch,feature|feature,hidden|batch,hidden"
fromEinsumString :: String -> ([Int], [Int], [Int])
fromEinsumString str = evalState build Map.empty
  where
  [sLhs,sRhs,sOut] = map (Misc.split ',') (Misc.split '|' str)
  build = do
    iLhs <- mapM toInt sLhs
    iRhs <- mapM toInt sRhs
    iOut <- mapM toInt sOut
    return (iLhs, iRhs, iOut)
  toInt :: String -> State (Map String Int) Int
  toInt s = do
    tableInn <- get
    case s `Map.lookup` tableInn of
      (Just i) -> return i
      Nothing -> do
        let i        = Map.size tableInn
            tableOut = Map.insert s i tableInn
        put tableOut
        return i



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

-- Invariants:
-- * every mergesplit and reblock must have one input and output
-- * there can be no [mergesplit or reblock] to [mergesplit or reblock] edges
_reblock :: Id -> BuildDagM Id
_reblock inn =
  let -- In this case, the given node is either a reblock or a merge. However,
      -- the invariant of one output might not be the case because mergesplit nodes
      -- can be dangling with 0 ouptuts waiting to be assigned the one output...
      fIsR (Node _ [actualInn] innDims ms@(MergeSplit _)) = do
        let newMergeSplitNode out = Node out [actualInn] innDims ms
        numOutsAtInn <- liftGraph $ IntSet.size <$> Graph.getUps inn
        case numOutsAtInn of
          -- this is a rogue mergesplit node being correctly used
          0 -> return inn
          -- this is a mergesplit node that has already been used;
          -- duplicate it getting another mergesplit node to use
          1 -> liftGraph $ Graph.insertObjectWithId newMergeSplitNode
          _ -> error "Too many outputs for a mergesplit node"
      fIsR (Node _ _ _ Reblock) =
        -- Assuming that the user did not create a rogue reblock node like one can do
        -- with merge and split nodes.
        error "calling _reblock with a reblock node is not allowed"

      -- In this case, the given node needs a reblock attached to it's output
      -- to use
      fIsNotR = do
        innDims <- getOutputDims inn
        let newReblockNode out = Node out [inn] innDims Reblock
        liftGraph $ Graph.insertObjectWithId newReblockNode
   in do node <- getObject inn
         if isReblockOrMergeSplit node
            then fIsR node
            else fIsNotR

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
    MergeSplit _ -> IntSet.empty

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

isReblockOrMergeSplit :: Node -> Bool
isReblockOrMergeSplit = _info .> f
  where f Reblock        = True
        f (MergeSplit _) = True
        f _              = False

assertNotMergeOrReblock :: Id -> BuildDagM ()
assertNotMergeOrReblock id = do
  node <- getObject id
  if isReblockOrMergeSplit node
     then error "this is a reblock or a mergesplit node"
     else return ()
