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
  ------------------------- Eq ( 3)
  xq   <- lift $ fBmm323 1.0 x wq   -- [h1,s,b],[h2,h1]->[h2,s,b]
  xqp  <- lift $ split nHH xq       -- [h2,s,b] -> [nHH, nN, nS, nB]
  xqpp <- lift $ transpose 1 2 xqp  -- [nHH, nN, nS, nB] -> [nHH, nS, nN, nB]

  ------------------------- Eq ( 5)
  xk    <- lift $ fBmm323 1.0 x wk       -- [nH,nS,nB]
  xkp   <- lift $ split nHH xk           -- [nHH,nN,nS,nB]
  xkppp <- lift $ permute [2,0,1,3] xkp  -- [nS,nHH,nN,nB]

  ------------------------- Eq ( 7)
  xv    <- lift $ fBmm323 1.0 x wv   -- [nH,nS,nB]
  xvp   <- lift $ split nHH xv       -- [nHH,nN,nS,nB]
  xvpp  <- lift $ transpose 1 2 xvp  -- [nHH,nS,nN,nB]

  ------------------------- Eq ( 9)
  x_score_p <- lift $ fBmm444 (1.0 / (sqrt (fromIntegral nHH))) xqpp xkppp -- [nB,nN,nS1,nS2]
  x_softmax <- lift $ softmax x_score_p                                    -- [nB,nN,nS1,nS2]

  ------------------------- Eq (11)
  x_att    <- lift $ fBmm444 1.0 x_softmax xvpp -- [nHH,nS,nN,nB]
  x_att_p  <- lift $ transpose 1 2 x_att        -- [nHH,nN,nS,nB]
  x_att_pp <- lift $ merge x_att_p              -- [nH,nS,nB]

  ------------------------- Eq (13)
  x_out   <- lift $ fBmm323 1.0 x_att_pp wo -- [nH,nS,nB]
  x_out_p <- lift $ add x_out x             -- [nH,nS,nB]

  ------------------------- Eq (15)
  x_mlp1   <- lift$ fBmm323 1.0 x_out_p w1          -- [4*nH,nS,nB]
  x_mlp1_p <- lift$ elementwise Relu [0,1,2] x_mlp1 -- [4*nH,nS,nB]

  ------------------------- Eq (17)
  x_mlp2   <- lift $ fBmm323 1.0 x_mlp1_p w2      -- [nH,nS,nB]
  x_mlp2_p <- lift $ add x_mlp2 x_out_p           -- [nH,nS,nB]

  -- RECURSE TO DO THE REST OF THE LAYERS
  g_x_mlp2_p <- compute layers x_mlp2_p           -- [nH,nS,nB]

  -- BACKWARD COMPUTATION
  ------------------------- Eq (18)
  let g_x_mlp2    = g_x_mlp2_p -- [nH,nS,nB]
  let g_x_out_p2  = g_x_mlp2_p -- [nH,nS,nB]
  g_x_mlp1_p <- lift $ fBmm323_ST 1.0 g_x_mlp2 w2       -- [4*nH, nS, nB]
  g_w2       <- lift $ contract_g_w2 x_mlp1_p g_x_mlp2  -- [nH, 4*nH]

  ------------------------- Eq (16)
  g_x_mlp1   <- lift $ elementwise Reluderiv [0,1,2] g_x_mlp1_p -- [4*nH, nS, nB]
  g_x_out_p1 <- lift $ contract_g_x_out_p1 g_x_mlp1 w1          -- [nH,nS,nB]
  g_x_out_p  <- lift $ add g_x_out_p1 g_x_out_p2                -- [nH,nS,nB]
  g_w1       <- lift $ contract_g_w1 x_out_p g_x_mlp1           -- [4*nH,nH]

  ------------------------- Eq (14)
  ------------------------- Eq (12)
  ------------------------- Eq (10)
  ------------------------- Eq ( 8)
  ------------------------- Eq ( 6)
  ------------------------- Eq ( 4)
  ------------------------- Eq ( 2)

  g_wq <- undefined
  g_wv <- undefined
  g_wk <- undefined
  g_wo <- undefined
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

-- [nH,nS,nB] [nH, four_nH] -> [four_nH,  nS, nB]
fBmm323_ST alpha = contractionAlpha [nH,nS,nB] [nH, fnH] [fnH, nS, nB] alpha
  where nH  = 0
        nS  = 1
        nB  = 2
        fnH = 3

-- 1h,h2->12
fBmm444 alpha =
  let hh = 0
      s1 = 1
      n  = 2
      b  = 3
      s2 = 4
      lhs = [hh,s1,n,b]
      rhs = [s2,hh,n,b]
      out = [s2,s1,n,b]
   in contractionAlpha lhs rhs out alpha

-- [fnH, nS, nB], [nH, nS, nB]
contract_g_w2 = contraction [fnH, nS, nB] [nH, nS, nB] [nH, fnH]
  where nH  = 0
        nS  = 1
        nB  = 2
        fnH = 3

contract_g_x_out_p1 = contraction [fnH, nS, nB] [fnH, nH] [nH, nS, nB]
  where nH  = 0
        nS  = 1
        nB  = 2
        fnH = 3

-- [nH,nS,nB] [fnH,nS,nB] -> [fnH,nH]
contract_g_w1 = contraction [nH,nS,nB] [fnH,nS,nB] [fnH,nH]
  where nH  = 0
        nS  = 1
        nB  = 2
        fnH = 3

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


