module BuildDag.Graph (
  Graph, Id, empty, singleton, map, mapWithKey, allIds, allObjects, allEdges,
  orderDag, orderDagItems, size,
  GraphM, getObject,
  getUps, getDowns, getAllDowns, putUps, putDowns, getLeafs, getRoots,
  deleteObject, insertObject, insertObjectWithId, updateObject,
  insertEdge, deleteEdge, isEdge,
  runGraphM, evalGraphM, execGraphM
) where

import Prelude hiding ( map )
import qualified Prelude as Prelude

import Data.Maybe

import Data.IntMap.Lazy ( IntMap )
import qualified Data.IntMap.Lazy as Map

import Data.IntSet ( IntSet, (\\), union, unions )
import qualified Data.IntSet as Set

import Control.Monad.Except
import Control.Monad.State
import Control.Monad.Identity

type Set = IntSet
type Map = IntMap

type GraphM a = ExceptT String (StateT (Graph a) Identity)
type Id = Int
type Ids = IntSet

-- Invariant: (i,j) means that i in down, and j in down[i].
--                             j in up,   adn i in up[j].
data Graph a = Graph {
  down    :: Map Ids,
  up      :: Map Ids,
  objects :: Map a,
  counter :: Int
}

empty :: Graph a
empty = Graph Map.empty Map.empty Map.empty 0

singleton :: a -> Graph a
singleton obj = Graph Map.empty Map.empty (Map.singleton 0 obj) 1

map :: (a -> b) -> Graph a -> Graph b
map f (Graph down up objects counter) =
  Graph down up (Map.map f objects) counter

mapWithKey :: (Id -> a -> b) -> Graph a -> Graph b
mapWithKey f (Graph down up objects counter) =
  Graph down up (Map.mapWithKey f objects) counter

-- This function is only guaranteed to behave correctly if the input
-- graph is indeed a Dag.
--
-- A node can be processed as long as every one of it's child nodes has
-- been processed. At the first iteration, all leaf nodes can be processed.
-- At the next iteration, more can be processed. An so on.
--
-- This function returns the the maximal nodes that can be proccessed at
-- each iteration. The first iteration is all leaf nodes.
-- The second is all nodes that can be processed after the leaf nodes.
-- And so on.
--
-- Note the return type: a list of sets, [Ids], not [Id].
orderDag :: Graph a -> [Ids]
orderDag d = evalGraphM orderDagM d

orderDagM :: GraphM a [Ids]
orderDagM =
  let recurse justProcessed processed = do
        -- get all of the items that might be cabable of being processed
        -- now
        allUps <- unions <$> mapM getUps (Set.toList justProcessed)
        -- an item can be processed if all of its downs are in processed
        let canProcessId name = (all (`Set.member` processed) . Set.toList) <$> getDowns name
        processThese <- Set.fromList <$> filterM canProcessId (Set.toList allUps)
        if Set.null processThese
           then return [] -- if nothing could be processed, assume we've processed
                          -- them all; the base case.
           else do theRest <- recurse processThese (processThese `union` processed)
                   return $ processThese : theRest
   in do
        leafs <- getLeafs
        if Set.null leafs
           then return []
           else do theRest <- recurse leafs leafs
                   return $ leafs:theRest

size :: Graph a -> Int
size = Map.size . objects

orderDagItems :: Graph a -> [(Id,a)]
orderDagItems d = flip evalGraphM d $ do
  names <- (concat . (Prelude.map Set.toList)) <$> (orderDagM :: GraphM a [Ids])
  objs  <- mapM getObject names
  return $ zip names objs

--  let recurse justProcessed processed = do
--        allUps <- unions <$> mapM getUps (Set.toList justProcessed)
--        let canProcess = Set.filter (`Set.member` processed) allUps
--        theRest <- recurse canProcess (canProcess `union` processed)
--        if Set.null canProcess
--           then return [] -- if nothing could be processed, assume we've processed
--                          -- them all; the base case.
--           else do theRest <- recurse canProcess (canProcess `union` processed)
--                   return $ canProcess : theRest
--
--      allUpsOf xs = unions <$> mapM getUps (Set.toList xs)
--   in flip evalGraphM d $ do
--        leafs <- getLeafs
--        if Set.null leafs
--           then return []
--           else do theRest <- recurse leafs leafs
--                   return $ leafs:theRest

allEdges :: Graph a -> [(Id, Id)]
allEdges (Graph down _ _ _) = Map.toList down |> Prelude.map fix |> concat
  where fix (above,belows) = Prelude.map (\below -> (above,below)) $ Set.toList belows

allIds :: Graph a -> Ids
allIds s = Map.keysSet $ objects s

allObjects :: Graph a -> Map a
allObjects = objects

hasId :: Id -> GraphM a Bool
hasId which = do
  cnt <- counter <$> get
  return (which < cnt)
-- hasId which = (isJust . (Map.lookup which) . objects) <$> get

hasIds :: [Id] -> GraphM a Bool
hasIds ids = do
  cnt <- counter <$> get
  return $ all (< cnt) ids
--  all id <$> mapM hasId ids

getObject :: Id -> GraphM a a
getObject which = do
  ((Map.! which) . objects) <$> get
--  objMaybe <- ((Map.lookup which) . objects) <$> get
--  case objMaybe of
--    Nothing -> throwError "getObject no id"
--    Just obj -> return obj

getUps :: Id -> GraphM a Ids
getUps which = requires (hasId which) "getUps no id" $ do
  up <- up <$> get
  case which `Map.lookup` up of
    Nothing -> return Set.empty
    Just ups -> return ups

getDowns :: Id -> GraphM a Ids
getDowns which = requires (hasId which) "getDowns no id" $ do
  down <- down <$> get
  case which `Map.lookup` down of
    Nothing -> return Set.empty
    Just downs -> return downs

getAllDowns :: Id -> GraphM a Ids
getAllDowns which = do
  downs <- getDowns which
  rest <- unions <$> mapM getAllDowns (Set.toList downs)
  return $ downs `union` rest

putDowns :: Id -> Ids -> GraphM a ()
putDowns which newDowns = requires (hasId which) "putDowns no id" $ do
  downTable <- down <$> get
  case which `Map.lookup` downTable of
   Nothing -> mapM_ (insertEdge which) $ Set.toList newDowns
   Just curDowns -> do
     let toDelete = curDowns \\ newDowns
         toInsert = newDowns \\ curDowns
     mapM_ (deleteEdge which) $ Set.toList toDelete
     mapM_ (insertEdge which) $ Set.toList toInsert

getLeafs :: GraphM a Ids
getLeafs = do
  allIds <- (Map.keys . objects) <$> get
  -- it's a leaf if the down is empty!
  let isLeaf id = Set.null <$> getDowns id
  Set.fromList <$> filterM isLeaf allIds

getRoots :: GraphM a Ids
getRoots = do
  allIds <- (Map.keys . objects) <$> get
  -- it's a leaf if the down is empty!
  let isRoot id = Set.null <$> getUps id
  Set.fromList <$> filterM isRoot allIds

putUps :: Id -> Ids -> GraphM a ()
putUps which newUps = requires (hasId which) "putUps no id" $ do
  upTable <- up <$> get
  case which `Map.lookup` upTable of
   Nothing -> mapM_ (flip insertEdge which) $ Set.toList newUps
   Just curUps -> do
     let toDelete = curUps \\ newUps
         toInsert = newUps \\ curUps
     mapM_ (flip deleteEdge which) $ Set.toList toDelete
     mapM_ (flip insertEdge which) $ Set.toList toInsert

updateObject :: Id -> a -> GraphM a ()
updateObject which object = requires (hasId which) "update no id" $ modify $ \s ->
  s { objects = Map.insert which object (objects s) }

deleteObject :: Id -> GraphM a ()
deleteObject which =
  let -- Here is what is going on
      --
      -- 1 ->   -> 5
      -- 2 -> 4 -> 6
      -- 3 ->
      --
      -- down[4] = 5,6.
      -- So delete down[4].
      -- But up[5] and up[6] still include 4.
      -- So go to up[5] an up[6] and adjust.
      --
      -- Now we have
      -- 1 ->
      -- 2 -> 4
      -- 3 ->
      -- down[4] = empty and no up[i] has 4.
      -- But up[4] = 1,2,3. So delete up[4]
      -- and go to down[1],down[2],down[3] and delete
      -- 4.
      --
      -- done.
      --
      -- This whole process is removeEdges.
      -- removeHelper is the part of this process that happens twice.
      removeHelper m0 n0 =
        case which `Map.lookup` m0 of
          Nothing -> (m0,n0)
          Just ss ->
            let m1 = Map.delete which m0
                n1 = foldr removeWhichOn n0 (Set.toList ss)
                removeWhichOn = Map.adjust (Set.delete which)
             in (m1,n1)
      removeEdges (m0,n0) =
        let (m1,n1) = removeHelper m0 n0
            (n2,m2) = removeHelper n1 m1
         in (m2,n2)
   in requires (hasId which) "delete no id" $ do
        Graph down0 up0 objects0 counter <- get
        let (down1, up1) = removeEdges (down0, up0)
            objects1 = Map.delete which objects0
        put $ Graph down1 up1 objects1 counter

getThenIncrementCounter :: GraphM a Id
getThenIncrementCounter = do
  cnt <- counter <$> get
  modify $ \s -> s { counter = cnt + 1 }
  return cnt

insertObjectWithId :: (Id -> a) -> GraphM a Id
insertObjectWithId toObj = do
  which <- getThenIncrementCounter
  let obj = toObj which
  modify $ \s -> s { objects = Map.insert which obj (objects s) }
  return which

insertObject :: a -> GraphM a Id
insertObject obj = insertObjectWithId (const obj)

insertEdge :: Id -> Id -> GraphM a ()
insertEdge above below =
  requires
    (hasIds [above,below])
    ("insertEdge requires both "++show above ++" and " ++ show below) $
    let addTo i j m =
          case i `Map.lookup` m of
            Nothing -> Map.insert i (Set.singleton j) m
            Just xs -> Map.insert i (Set.insert j xs) m
     in modify $ \s -> s { down = addTo above below (down s),
                           up   = addTo below above (up s) }

deleteEdge :: Id -> Id -> GraphM a ()
deleteEdge above below =
  let delete i j m =
        case i `Map.lookup` m of
          Nothing -> m
          Just xs -> Map.insert i (Set.delete j xs) m
   in modify $ \s -> s { down = delete above below (down s),
                         up   = delete below above (up s) }

isEdge :: Id -> Id -> GraphM a Bool
isEdge above below = do
  down <- down <$> get
  case above `Map.lookup` down of
    Nothing -> return False
    Just ds -> return (below `Set.member` ds)

runGraphM :: GraphM a b -> Graph a -> (b, Graph a)
runGraphM doThis gInit =
  case runExceptT doThis |> (`runStateT` gInit) |> runIdentity of
    (Left err, _)      -> error err
    (Right bOut, gOut) -> (bOut, gOut)

(|>) = flip ($)

evalGraphM :: GraphM a b -> Graph a -> b
evalGraphM doThis gInit = fst $ runGraphM doThis gInit

execGraphM :: GraphM a b -> Graph a -> Graph a
execGraphM doThis gInit = snd $ runGraphM doThis gInit

-- This function is for debugging and can be quite costly.
-- It is turned off for now
requires :: GraphM a Bool -> String -> GraphM a b -> GraphM a b
requires isGood errMsg doThis = do
  doThis
--  canContinue <- isGood
--  if canContinue
--     then doThis
--     else throwError errMsg


