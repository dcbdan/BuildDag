module BuildDag.Types(
  Id, Rank, Dim, Dims,
  NodeInfo(..), Node(..), Dag, DagM, BuildDagM,
  Kernel(..), Param(..),  Paramable(..),
  UOp(..), BOp(..), CastableBOp(..), Init(..),
  ---
  checkContraction, incToOut, incToAgg
) where

import Data.IntSet ( IntSet, (\\) )
import qualified Data.IntSet as IntSet

import Data.Map ( Map, (!) )
import qualified Data.Map as Map

import BuildDag.Graph ( Graph, GraphM )
import qualified BuildDag.Graph as Graph

import Data.List ( sortOn )

import Control.Monad.RWS ( RWS )

import BuildDag.Misc ( sepBy, idxInterval, fst3, (.>), (|>) )

type Id = Graph.Id

type Dim = Int
type Dims = [Dim]
type Rank = Int

data NodeInfo =
    Input Init
  | Join Kernel
  | Reblock
  | Agg CastableBOp

instance Paramable NodeInfo where
  paramable = f
    where
    f (Input init)  = (Pi 0):(paramable init)
    f (Join kernel) = (Pi (whichKI kernel)):(paramable kernel)
    f (Reblock)     = [Pi 1]
    f (Agg op)      = (Pi 7):(paramable op)
    whichKI (KI_Contraction  _ _ _ _  ) = 2
    whichKI (KI_Reduction    _ _ _ _  ) = 3
    whichKI (KI_EW           _ _ _    ) = 4
    whichKI (KI_EWB          _ _ _ _ _) = 5
    whichKI (KI_Dropout      _ _      ) = 6

data Node = Node {
  _id      :: Id,
  _inns    :: [Id],
  _incDims :: Dims,
  _info    :: NodeInfo
}

type Dag = Graph Node
type DagM = GraphM Node

type BuildDagM = RWS
  (Map String Dims) -- A lookup table for dimensions of the inputs
  ()                -- There is nothing to write
  Dag               -- The state is a dag

data Kernel =
    KI_Contraction [Int] [Int] [Int] Float
  | KI_Reduction CastableBOp Int IntSet Float
  | KI_EW UOp [Int] Float
  | KI_EWB BOp [Int] [Int] Float Float
  | KI_Dropout Int Float

data Param = Pi Int | Pf Float | Pb Bool

-- what are the params of this object?
class Paramable p where
  paramable :: p -> [Param]

data CastableBOp =
    CastableAdd
  | CastableMax
  | CastableMin

instance Paramable CastableBOp where
  paramable CastableAdd = paramable Add
  paramable CastableMax = paramable Max
  paramable CastableMin = paramable Min

data BOp =
    Add
  | Max
  | Min
  | Mul
  | Sub
  | Div

instance Paramable BOp where
  paramable Add = [Pi 0]
  paramable Max = [Pi 1]
  paramable Min = [Pi 2]
  paramable Mul = [Pi 3]
  paramable Sub = [Pi 4]
  paramable Div = [Pi 5]

instance Show BOp where
  show Add = "add"
  show Max = "max"
  show Min = "min"
  show Mul = "mul"
  show Sub = "sub"
  show Div = "div"

data UOp =
    Sigmoid
  | Exp
  | Square
  | Relu
  | Reluderiv
  | Sqrt
  | AddScalar Float

instance Paramable UOp where
  paramable Sigmoid       = [Pi 0]
  paramable Exp           = [Pi 1]
  paramable Square        = [Pi 2]
  paramable Relu          = [Pi 3]
  paramable Reluderiv     = [Pi 4]
  paramable Sqrt          = [Pi 5]
  paramable (AddScalar f) = [Pi 6, Pf f]

instance Show UOp where
  show  Sigmoid      = "sigmoid"
  show  Exp          = "exp"
  show  Square       = "square"
  show  Relu         = "relu"
  show  Reluderiv    = "reluderiv"
  show  Sqrt         = "sqrt"
  show (AddScalar _) = "addScalar"


data Init =
    InitConstant Float
  | InitRandom   Float Float
  | InitFile     Int

instance Paramable Init where
  paramable (InitConstant val)    = [Pi 1, Pf val]
  paramable (InitRandom low high) = [Pi 0, Pf low, Pf high]
  paramable (InitFile whichFile)  = [Pi 2, Pi whichFile]

instance Paramable Kernel where
  paramable (KI_Contraction lhs rhs out alpha) =
    nl:nr:no:(map Pi lhs ++ map Pi rhs ++ map Pi out ++ [Pf alpha])
    where
      nl = Pi (length lhs)
      nr = Pi (length rhs)
      no = Pi (length out)
  paramable (KI_Reduction op _ outSet alpha) = concat [paramable op, [Pf alpha], map Pi out]
    where out = IntSet.toAscList outSet
  paramable (KI_EW uop out alpha) = concat [paramable uop, [Pf alpha], map Pi out]
  paramable (KI_EWB bop lhs rhs alpha beta) = concat [
    paramable bop,
    [Pf alpha, Pf beta],
    (Pi (length lhs)):(map Pi lhs),
    (Pi (length rhs)):(map Pi rhs)]
  paramable (KI_Dropout _ f) = [Pf f]

checkContraction [] [] [] = False -- don't do this
checkContraction lhs rhs out =
  let inInput m = (m `elem` lhs) || (m `elem` rhs)
      i = maximum (lhs ++ rhs)
   in (all inInput out) && (all inInput [0..i])

incToOut :: IntSet -> [a] -> [a]
incToOut aggd inc = map fst $ filter isNotAgg $ zip inc [0..]
  where
  isAgg    (_, idx) = idx `IntSet.member` aggd
  isNotAgg = not . isAgg

incToAgg :: IntSet -> [a] -> [a]
incToAgg aggd inc = map fst $ filter isAgg $ zip inc [0..]
  where
  isAgg    (_, idx) = idx `IntSet.member` aggd


