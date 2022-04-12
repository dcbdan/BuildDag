import System.Environment ( getProgName, getArgs )

import BuildDag.Dags    ( parseDagArgs )
import BuildDag.ShowDag ( showDag )
import BuildDag.Types   ( Dag )

import BuildDag.Misc ( sepBy )

main :: IO ()
main  = do
  progName <- getProgName
  args <- getArgs
  case parseArgs args of
    Nothing -> do
      mapM_ putStrLn (usage progName)
    Just dag -> do
      putStrLn $ sepBy " " id args
      putStr   $ unlines $ showDag dag

usage progName = ["Usage: " ++ progName ++ " DagName DagArgs"]

parseArgs :: [String] -> Maybe Dag
parseArgs (dagIdentifier:dagArgs) =
  parseDagArgs dagIdentifier dagArgs
parseArgs _ = Nothing

