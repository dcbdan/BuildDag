module BuildDag.ShowDag( showDag ) where

import qualified Data.IntSet as IntSet
import qualified Data.IntMap as IntMap

import BuildDag.Types
import BuildDag.Kernel
import BuildDag.Misc ( sepBy )
import qualified BuildDag.Graph as Graph

-- matmul 10032 10032 10032
-- I[i0i0f-1.0f1.0]|10032,10032
-- I[i0i0f-1.0f1.0]|10032,10032
-- R[i1]0|10032,10032
-- R[i1]1|10032,10032
-- J[i2i2i2i2i0i1i1i2i0i2f1.0]2,0,1$3,1,2:1|10032,10032,10032
-- A[i7i0]4|10032,10032

instance Show Node where
  show = show_
    where
      show_ (Node id inns dims ni) = f ni inns  ++ "|" ++ sepBy "," show dims
      f ni@(Input init) [] = "I" ++ showParams (paramable ni)
      f ni@(Join kernel) inns = "J" ++ showParams (paramable ni) ++ cStr ++ ":" ++ aggStr
        where aggStr = sepBy "," show (IntSet.toList (getAggRanks kernel))
              cStr = sepBy "$" showC $ zip inns (getOrderings kernel)
              showC (id,ranks) = sepBy "," show (id:ranks)
      f ni@(Reblock) [inn] = "R" ++ showParams (paramable ni) ++ show inn
      f ni@(Agg op) [inn] = "A" ++ showParams (paramable ni) ++ show inn
      f ni@(MergeSplit ms) [inn] = "M" ++ showParams (paramable ni) ++ show inn
      f _ _ = error "Incorrect number of inputs given to show for Node"

      showParams params = "[" ++ sepBy "" showParam params ++ "]"
      showParam (Pi i)     = "i"++show i
      showParam (Pf f)     = "f"++show f
      showParam (Pb True)  = "b1"
      showParam (Pb False) = "b0"

showDag :: Dag -> [String]
showDag dag = map (show . snd) $ IntMap.toAscList (Graph.allObjects dag)

