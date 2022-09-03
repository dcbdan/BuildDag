module BuildDag.Dags.BHT ( -- BHT == BinHang's Transformer
  Params(..), bht
) where

import Prelude hiding ( subtract )

import Data.Map ( Map )
import qualified Data.Map as Map

import BuildDag.Types
import BuildDag.Build

import Control.Monad.State

data Params = Params {
    _nB  :: Int,
    _nS  :: Int,
    _nH  :: Int,
    _nN  :: Int,
    _nHH :: Int
  }

data BhtState = BhtState {
    bht_ids                :: Map String Id,
    bht_params             :: Params,
    bht_learning_rate      :: Float,
    bht_compute_error_grad :: Id -> BuildDagM Id
  }

type BhtM = StateT BhtState BuildDagM

bht :: Int -> Params -> (Map String Dims, BuildDagM ())
bht numLayers params@(Params nB nS nH nN nHH) =
  let inputTable = Map.fromList $
        ("x", [nH,nS,nB]):("y", [nH,nS,nB]):(concat (map fParams layerNames))
      fParams layerName = [
        (paramName layerName "wq", [nH,nH]),
        (paramName layerName "wv", [nH,nH]),
        (paramName layerName "wk", [nH,nH]),
        (paramName layerName "wo", [nH,nH]),
        (paramName layerName "w1", [4*nH, nH]),
        (paramName layerName "w2", [nH, 4*nH])]
      layerNames = map show [1..numLayers]

      initState = BhtState
        Map.empty
        params
        learningRate
        softmaxCrossEntropyWithLogitsGrad
      learningRate = 0.0001
      leastSquareGrad yhat = do
        y <- initRandom "y" (-1.0) (1.0)
        -- d/dyhat(i) sum_j (yhat(j)-y(j))^2 = 2 * (yhat(i) - y(i))
        --                             approx= yhat(i) - y(i)
        subtract yhat y
      softmaxCrossEntropyWithLogitsGrad logits = do
        -- TODO(wish): the labels should really be a probability distribution
        --             along the leading dimension, but whatever
        labels <- initRandom "y" 0.0 0.001
        -- From TensorFlow:
        --   log_probs = log_softmax_v2(precise_logits)
        --   cost = -math_ops.reduce_sum(labels * log_probs, axis=1)
        --
        -- After doing the gradient...
        --   dcost / d x(i) = l_i - (sum_j l(j)) * e^x(i) / (sum_j e^x(i))
        let x = logits
            l = labels
            allR = [0,1,2]
            lstR = [1,2]
        ex    <- elementwise Exp allR x
        sumEx <- reduction CastableAdd lstR ex
        sumL  <- reduction CastableAdd lstR l
        ediv  <- elementwiseBinary Div allR lstR allR ex sumEx
        lediv <- elementwiseBinary Mul allR lstR allR ediv sumL
        grad  <- subtract l lediv
        return grad

      buildDag = flip evalStateT initState $ do
        x <- lift $ initRandom "x" (-0.001) (0.001)
        grad_x <- compute layerNames x
        return ()

   in (inputTable, buildDag)

--inputs (Params nB nS nH nN nHH) = Map.fromList [
--  ("x", [nH,nS,nB]),
--  ("wq", [nH,nH]),
--  ("wv", [nH,nH]),
--  ("wk", [nH,nH]),
--  ("wo", [nH,nH]),
--  ("w1", [4*nH, nH]),
--  ("w2", [nH, 4*nH])]

-- If we have prefix / name, then return the id;
-- otherwise, initRandom and get
getParam :: String -> String -> BhtM Id
getParam prefix name = do
  table <- bht_ids <$> get
  let str = paramName prefix name
  case str `Map.lookup`table of
    (Just id) -> return id
    Nothing -> do id <- lift $ initRandom str (-1.0) (1.0)
                  state <- get
                  modify $ \state -> state{ bht_ids = Map.insert str id table }
                  return id

paramName prefix name = prefix ++ "/" ++ name

descentParam :: String -> String -> Id -> BhtM ()
descentParam prefix name grad = do
  let str = paramName prefix name
  vOld        <- (flip (Map.!) str) . bht_ids <$> get
  lr          <- bht_learning_rate <$> get
  scaled_grad <- lift $ scale lr grad
  vNew        <- lift $ subtract vOld scaled_grad
  -- ^ vNew = vOld - lr * grad
  modify $ \state -> state { bht_ids = Map.insert str vNew (bht_ids state) }

compute :: [String] -> Id -> BhtM Id
compute [] yhat = do
  f <- bht_compute_error_grad <$> get
  lift $ f yhat

compute (thisLayer:layers) x = do
  params <- bht_params <$> get
  let Params nB nS nH nN nHH = params

  wq <- getParam thisLayer "wq"
  wv <- getParam thisLayer "wv"
  wk <- getParam thisLayer "wk"
  wo <- getParam thisLayer "wo"
  w1 <- getParam thisLayer "w1"
  w2 <- getParam thisLayer "w2"

  -- FORWARD COMPUTATION
  xq   <- lift $ fBmm323 1.0 x wq   -- [h1,s,b],[h2,h1]->[h2,s,b]
  xqp  <- lift $ split nHH xq       -- [h2,s,b] -> [nHH, nN, nS, nB]
  xqpp <- lift $ transpose 1 2 xqp  -- [nHH, nN, nS, nB] -> [nHH, nS, nN, nB]

  xk    <- lift $ fBmm323 1.0 x wk       -- [nH,nS,nB]
  xkp   <- lift $ split nHH xk           -- [nHH,nN,nS,nB]
  xkppp <- lift $ permute [2,0,1,3] xkp  -- [nS,nHH,nN,nB]

  xv    <- lift $ fBmm323 1.0 x wv   -- [nH,nS,nB]
  xvp   <- lift $ split nHH xv       -- [nHH,nN,nS,nB]
  xvpp  <- lift $ transpose 1 2 xvp  -- [nHH,nS,nN,nB]

  x_score_p <- lift $ fBmm444 (1.0 / (sqrt (fromIntegral nHH))) xqpp xkppp -- [nB,nN,nS1,nS2]
  x_softmax <- lift $ softmax x_score_p                                    -- [nB,nN,nS1,nS2]

  x_att    <- lift $ fBmm444 1.0 x_softmax xvpp -- [nHH,nS,nN,nB]
  x_att_p  <- lift $ transpose 1 2 x_att        -- [nHH,nN,nS,nB]
  x_att_pp <- lift $ merge x_att_p              -- [nH,nS,nB]

  x_out   <- lift $ fBmm323 1.0 x_att_pp wo -- [nH,nS,nB]
  x_out_p <- lift $ add x_out x             -- [nH,nS,nB]

  x_mlp1   <- lift$ fBmm323 1.0 x_out_p w1          -- [4*nH,nS,nB]
  x_mlp1_p <- lift$ elementwise Relu [0,1,2] x_mlp1 -- [4*nH,nS,nB]

  x_mlp2   <- lift $ fBmm323 1.0 x_mlp1_p w2      -- [nH,nS,nB]
  x_mlp2_p <- lift $ add x_mlp2 x_out_p           -- [nH,nS,nB]

  -- RECURSE TO DO THE REST OF THE LAYERS
  g_x_mlp2_p <- compute layers x_mlp2_p

  -- BACKWARD COMPUTATION

  g_wq <- undefined
  g_wv <- undefined
  g_wk <- undefined
  g_wo <- undefined
  g_w1 <- undefined
  g_w2 <- undefined
  g_x  <- undefined

  -- DO THE GRADIENT STEP
  descentParam thisLayer "wq" g_wq
  descentParam thisLayer "wv" g_wv
  descentParam thisLayer "wk" g_wk
  descentParam thisLayer "wo" g_wo
  descentParam thisLayer "w1" g_w1
  descentParam thisLayer "w2" g_w2

  -- RETURN UPDATED VALUE
  return g_x

-------------------------------------------------------------------------------

fBmm323 alpha = contractionAlpha [2,1,0] [3,2] [3,1,0] alpha

-- 1h,h2->12
fBmm444 alpha innL innR =
  let hh = 0
      s1 = 1
      n  = 2
      b  = 3
      s2 = 4
      lhs = [hh,s1,n,b]
      rhs = [s2,hh,n,b]
      out = [s2,s1,n,b]
   in contractionAlpha lhs rhs out alpha innL innR

-- This is softmaxing the leading dimension, dimension 0.
--
-- softmax(x)_i = e^(x_i - c)/(\sum_i (e^(x_i - c)))
-- where c = max_i x_i
-- (the computation is the same for all c, but numerically more
--  stable when the largest value of x is zero)
softmax :: Id -> BuildDagM Id
softmax x = do
  n <- length <$> getOutputDims x
  let allR = [0..(n-1)]
      lstR = [1..(n-1)]
  -- x                                                      ij
  c   <- reduction CastableMax lstR x                    -- j
  xc  <- elementwiseBinary Sub allR lstR allR x c        -- ij
  ex  <- elementwise Exp allR xc                         -- ij
  de  <- reduction CastableAdd  lstR ex                  -- j
  out <- elementwiseBinary Div allR lstR allR ex de      -- ij
  return out


