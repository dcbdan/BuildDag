module BuildDag.Misc(
    remove, commaSepListDot, unwordsS, setEqual, subsetEqual, isUnique
  , filterOn, filterWhich
  , allCombos, allIntCombos, sortThenGroupBy, nSquaredGroupBy, (.>), (|>), sepBy
  , mapTuple, mapFst, mapSnd, flipTup, fst3, snd3, trd3
  , eMapToLeftList, eMapToRightList, mapIntersectSet, setToMap
  , listUpdate, insertSortedList, findMin, findArgMin, findBestN, idxInterval
  , formatTable, printTable
)where

import Data.Maybe
import Data.Either
import Data.List
import Data.Function (on)

import Data.Map ( Map )
import qualified Data.Map as Map

import Data.Set ( Set, member )
import qualified Data.Set as Set

-- like filter, but only remove the first and negated
remove :: (a -> Bool) -> [a] -> [a]
remove predicate [] = []
remove predicate (x:xs) | predicate x = xs
remove predicate (x:xs)               = x:(remove predicate xs)

commaSepListDot :: [String] -> String
commaSepListDot [] = ""
commaSepListDot (x:[]) = x ++ " . "
commaSepListDot (x:xs) = x ++ ", " ++ commaSepListDot xs

unwordsS :: (a -> String) -> String -> [a] -> String
unwordsS _ _ [] = ""
unwordsS toStr _ (x:[]) = toStr x
unwordsS toStr sepWith (x:xs) = toStr x ++ sepWith ++ unwordsS toStr sepWith xs

setEqual :: Eq a => [a] -> [a] -> Bool
setEqual xs ys = null (xs \\ ys) && null (ys \\ xs)

-- check if all elements in one set is in the other
subsetEqual :: Eq a => [a] -> [a] -> Bool
subsetEqual xs ys = all isInYs xs
    where isInYs x = isJust $ find (==x) ys

isUnique :: Eq a => [a] -> Bool
isUnique xs = (length (nub xs)) == (length xs)

filterOn :: (a -> b) -> (b -> Bool) -> [a] -> [a]
filterOn f b xs = filter (b . f) xs

-- input:  [[1,2], [4,5]]
-- output: [[1,4],[1,5],[2,4],[2,5]]
allCombos :: [[a]] -> [[a]]
allCombos [] = []
allCombos as = recurse as
  where recurse [] = [[]]
        recurse (x:xs) = do
          y <- x
          ys <- recurse xs
          return (y:ys)

-- TODO: if 0, should be empty? i.e., shouldnt it be the case that
--       (product x) == length (allIntCombos x)  ?
-- [0,1,2] -> [[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2]]
-- [] -> [[]]
allIntCombos :: [Int] -> [[Int]]
allIntCombos [] = [[]]
allIntCombos is = allCombos $ map (\i -> (idxInterval i)) is

--sortThenGroupBy f compare xs = sortOn f xs >>= groupBy compare
sortThenGroupBy :: Ord b => (a -> b) -> [a] -> [[a]]
sortThenGroupBy f xs =
   let xys = map (\a -> (a, f a)) xs
       xysSorted = sortOn snd xys
       xysGrouped = groupBy ((==) `on` snd) xysSorted
       xsGrouped = map (map fst) xysGrouped
    in xsGrouped

-- Here is an n^2 algorithm grouping all common items.
-- toB is applied n times.
nSquaredGroupBy :: Eq b => (a -> b) -> [a] -> [[a]]
nSquaredGroupBy toB as =
  let abs = map (\a -> (a, toB a)) as
      f [] = []
      f ((a,b):xs) =
        let (thisGroup, theRest) = partition (snd .> (== b)) xs
         in ((a,b):thisGroup):(f theRest)
   in f abs |> map (map fst)

(.>) = flip (.)
(|>) = flip ($)

sepBy :: String -> (a -> String) -> [a] -> String
sepBy comma toStr [] = ""
sepBy comma toStr (x:[]) = toStr x
sepBy comma toStr (x:xs) = toStr x ++ comma ++ sepBy comma toStr xs

eMapToLeftList  :: Map k (Either a b) -> [(k,a)]
eMapToLeftList m =
  let mLeft = Map.filter isLeft m
      (keys, lefts) = unzip $ Map.toList mLeft
      as = map (fromLeft undefined) lefts
   in zip keys as

eMapToRightList :: Map k (Either a b) -> [(k,b)]
eMapToRightList m =
  let mRight = Map.filter isRight m
      (keys, rights) = unzip $ Map.toList mRight
      bs = map (fromRight undefined) rights
   in zip keys bs

mapTuple :: (a -> b) -> (a,a) -> (b,b)
mapTuple f (x,y) = (f x, f y)

mapFst :: (a -> c) -> [(a, b)] -> [(c, b)]
mapFst f = map (\(x,y) -> (f x, y))

mapSnd :: (b -> c) -> [(a, b)] -> [(a, c)]
mapSnd f = map (\(x,y) -> (x, f y))

flipTup :: (a,b) -> (b,a)
flipTup (x,y) = (y,x)

fst3 :: (a,b,c) -> a
fst3 (x,_,_) = x

snd3 :: (a,b,c) -> b
snd3 (_,y,_) = y

trd3 :: (a,b,c) -> c
trd3 (_,_,z) = z

mapIntersectSet :: Ord k => Map k a -> Set k -> Map k a
mapIntersectSet m s = Map.filterWithKey isInSet m
  where isInSet k _ = k `member` s

-- 1) Convert set to ascending list O(n)
-- 2) Convert ascending  list to map O(n)
-- 3) apply a map to each key to obtain the resulting value O(n)
setToMap :: Eq k => (k -> a) -> Set k -> Map k a
setToMap f = Set.toAscList .> makeKeysUnits .> Map.fromAscList .> Map.mapWithKey f'
  where makeKeysUnits xs = zip xs $ repeat ()
        f' k _ = f k

-- this is not an efficient function!
-- lists shouldn't be used this way...
listUpdate :: Int -> a -> [a] -> [a]
listUpdate i val = zip [0..] .> map fix
  where fix (j, x) | j == i = val
        fix (j, x)          = x

-- this is also not an efficient fuction!
-- lists shouldn't be used this way
insertSortedList :: Ord a => a -> [a] -> [a]
insertSortedList val [] = [val]
insertSortedList val (x:xs) | val < x = val:x:xs
insertSortedList val (x:xs) = x:(insertSortedList val xs)

findMin :: Ord b => (a -> b) -> [a] -> a
findMin _ [] = error "findMin cannot have an empty-list input"
findMin f xs = map f' xs |> foldr1 g |> snd
  where f' x = (f x, x)
        g r1@(v1,_) r2@(v2,_) = if v1 < v2 then r1 else r2

findArgMin :: Ord b => (a -> b) -> [a] -> Int
findArgMin f xs =
  let f' (_, x) = f x
   in fst $ findMin f' $ zip [0..] xs

findBestN :: Ord b => Int -> (a -> b) -> [a] -> [a]
findBestN n f xs = take n $ sortBy (compare `on` f) xs

idxInterval :: Int -> [Int]
idxInterval i | i < 1 = error "idxInterval index less than 1"
idxInterval i = [0..(i-1)]

formatTable :: Int -> [[String]] -> [String]
formatTable sepSize table = map formatRow table
  where
  maxByColumn = transpose table |> map (map length .> maximum)
  formatRow = concat . (map (uncurry f)) . (zip maxByColumn)
    where f n x = x ++ (replicate (sepSize + n - length x) ' ')

printTable :: Int -> [[String]] -> IO ()
printTable sepSize = formatTable sepSize .> mapM_ putStrLn

filterWhich :: (Int -> Bool) -> [a] -> [a]
filterWhich f xs = zip [0..] xs |> filter (fst .> f) |> map snd


