module BuildDag.Module(
  Module(..), Name,
  initialize, perturb, buildForward,
  forward11, forward_1, forward1_, forward__,
  variablesMapping, tensorsMapping, modelSize,
  rowMajorLinearModule, generalLinearModule,
  straightForward
) where

import Prelude hiding ( subtract )

import Control.Monad ( foldM )

import Data.Map ( Map )
import qualified Data.Map as Map

import BuildDag.Types
import BuildDag.Build

import BuildDag.Misc ( (|>), (.>), formatTable, sepBy )

-- Copying https://github.com/codertimo/BERT-pytorch
-- They have modules, modules have initialization and
-- forward functions. Do that here.
--
-- Every module has variables that need to be initialized and
-- a forward function that maps those initialized variables and
-- some inputs to an output
data Module a =
    Module {
      name       :: String,
      moduleVars :: [Module a],
      -- ^ the list of all training tensors and modules that this particular module
      --   requires for the forward computation
      forward :: [Module Id] -> [Id] -> BuildDagM [Id]
      -- ^ given training information, input information that match init and inputs,
      --   but with ids instead of tensors, construct the outputs but with ids
    }
  | Tensor Bool String Dims Init a
  -- The Bool specifies that the tensor is a variable and not a constant..

type Name = [String]

forward11 mod x  = head <$> forward mod (moduleVars mod) [x]
forward_1 mod xs = head <$> forward mod (moduleVars mod) xs
forward1_ mod x  =          forward mod (moduleVars mod) [x]
forward__ mod xs =          forward mod (moduleVars mod) xs

-- initialize tensors and create a new module containing
-- the ids of the training variables in the dag
initialize :: Module () -> BuildDagM (Module Id)
initialize = f []
  where
  f nms (Tensor isVar name dims init ()) =
    Tensor isVar name dims init <$> initInput fullName init
      where fullName = fullNameOf (name:nms)
  f nms mod = do
    vars <- mapM (f (name mod:nms)) (moduleVars mod)
    return $ mod { moduleVars = vars }

-- For each variable, create an identical tensor and subtract from that
perturb ::
  Float -> Float -> (String -> String) ->
  Module Id -> BuildDagM (Module Id)
perturb a b rename = recurse []
  where
  recurse nms (Tensor True name dims init tInn) = do
    let fullName = fullNameOf (name:nms)
    perturbed <- initRandom (rename fullName) a b
    tOut <- subtract tInn perturbed
    return $ Tensor True name dims init tOut
  recurse _ t@(Tensor False _ _ _ _) = return t
  recurse nms (Module name varsInn forward) = do
    varsOut <- mapM (recurse (name:nms)) varsInn
    return $ Module name varsOut forward

-- given a module with labeled ids, map input ids to output ids
buildForward :: Module Id -> [Id] -> BuildDagM [Id]
buildForward (Module _ vars forward) = forward vars
buildForward (Tensor _ _ _ _ _) = error "tensors cannot be forwarded"

variablesOf :: Module a -> [(Name, Dims, a)]
variablesOf = f []
  where
  f nms (Tensor True name dims init v) = [(name:nms, dims, v)]
  f nms (Tensor False _ _ _ _) = []
  f nms (Module name vars _) = concat $ map (f (name:nms)) vars

tensorsOf :: Module a -> [(Name, Dims, a)]
tensorsOf = f []
  where
  f nms (Tensor _ name dims init v) = [(name:nms, dims, v)]
  f nms (Module name vars _) = concat $ map (f (name:nms)) vars

instance Show a => Show (Module a) where
  show = tensorsOf .> toTable .> formatTable 2 .> unlines
    where toTable tvars =
            let largestNameSpace = maximum $ map (fst3 .> length) tvars
                fst3 (x,_,_) = x
                fix (namespace, dims, v) =
                  let namespace_ = reverse $
                        namespace ++
                        replicate (largestNameSpace - length namespace) ""
                   in namespace_ ++ [show dims] ++ [show v]
             in map fix tvars

variablesMapping :: Module a -> Map String Dims
variablesMapping = variablesOf .> map fix .> Map.fromList
  where fix (name, dims, _) = (fullNameOf name, dims)

tensorsMapping :: Module a -> Map String Dims
tensorsMapping = tensorsOf .> map fix .> Map.fromList
  where fix (name, dims, _) = (fullNameOf name, dims)

modelSize :: Module a -> Int
modelSize = variablesMapping .> Map.elems .> map product .> sum

-- This is name rowMajorLinearModule and not linearModule as a reminder
-- that the batch dimension i is in the leading dimension only for row major
-- computations.
--
-- "ij,jk->ik"
-- In this case,
--   W is jk,
--   x is ij,
--   b is k
-- and the computation is
--   xW + b
rowMajorLinearModule :: String -> Dim -> Dim -> Dim -> Module ()
rowMajorLinearModule name nI nJ nK = Module name linearVars linearForward
  where
  linearVars = [
    Tensor True "W" [nJ, nK] (InitRandom (-1.0) 1.0) (),
    Tensor True "b" [nK]     (InitRandom (-1.0) 1.0) ()]
  linearForward [Tensor _ _ _ _ w, Tensor _ _ _ _ b] [x] = do
    xw <- matmul x w
    --ik,k->ik ----------         i k   k   i k
    xwb <- elementwiseBinary Add [0,1] [1] [0,1] xw b
    return [xwb]

generalLinearModule :: String -> [Int] -> [(Dim, Int)] -> [Int] -> Module ()
generalLinearModule name inModes paramInfo outModes = Module name linearVars linearForward
  where
  -- aggModes  = modes     in inModes and not in outModes  |- in
  -- keepModes = modes     in inModes and     in outModes  |-     |-
  -- newModes  = modes not in inModes and     in outModes         |- w  |- b
  (wDims, wModes) = unzip paramInfo
  (bDims, bModes) = unzip $ filter isNewMode paramInfo
    where isNewMode (dim,mode) =
            (not $ mode `elem` inModes) && (mode `elem` outModes)
  linearVars = [
    Tensor True "W" wDims (InitRandom (-1.0) 1.0) (),
    Tensor True "b" bDims (InitRandom (-1.0) 1.0) ()]
  linearForward [Tensor _ _ _ _ w, Tensor _ _ _ _ b] [x] = do
    xw  <- contraction inModes wModes outModes x w
    xwb <- elementwiseBinary Add outModes bModes outModes xw b
    return [xwb]

-- Each of the modules has one input and one output.
-- Do one after the other to get the output.
straightForward :: [Module Id] -> [Id] -> BuildDagM [Id]
straightForward modules [xIn] = do
  let apply = flip forward11
  xOut <- foldM apply xIn modules
  return [xOut]

fullNameOf :: Name -> String
fullNameOf = reverse .> sepBy ":" id

