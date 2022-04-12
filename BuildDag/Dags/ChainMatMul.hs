module BuildDag.Dags.ChainMatMul( 
  BinaryTree(..), 
  chainMatMul  
  --randomTrees TODO
) where

import BuildDag.Types
import BuildDag.Build

data BinaryTree =
    CMLeaf String
  | CMNode BinaryTree BinaryTree

chainMatMul :: BinaryTree -> BuildDagM Id
chainMatMul (CMLeaf name) = initRandom name (-1.0) 1.0
chainMatMul (CMNode lhs rhs) = do
  lhsId <- chainMatMul lhs
  rhsId <- chainMatMul rhs
  matmul lhsId rhsId

