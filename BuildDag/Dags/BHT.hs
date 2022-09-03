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
  xq   <- lift $ einsum "h1,s,b|h2,h1|h2,s,b" x wq -- [h2,s,b]
  xqp  <- lift $ split nHH xq                      -- [nHH, nN, nS, nB]
  xqpp <- lift $ transpose 1 2 xqp                 -- [nHH, nS, nN, nB]

  ------------------------- Eq ( 5)
  xk    <- lift $ einsum "h1,s,b|h2,h1|h2,s,b" x wk -- [h2,s,b]
  xkp   <- lift $ split nHH xk                      -- [nHH,nN,nS,nB]
  xkppp <- lift $ permute [2,0,1,3] xkp             -- [nS,nHH,nN,nB]

  ------------------------- Eq ( 7)
  xv    <- lift $ einsum "h1,s,b|h2,h1|h2,s,b" x wv -- [h2,s,b]
  xvp   <- lift $ split nHH xv                      -- [nHH,nN,nS,nB]
  xvpp  <- lift $ transpose 1 2 xvp                 -- [nHH,nS,nN,nB]

  ------------------------- Eq ( 9)
  x_score_p <- lift $
    let alpha = (1.0 / (sqrt (fromIntegral nHH)))
     in fBmm444_SS_Alpha alpha xqpp xkppp -- [s2,s1,n,b]
  x_softmax <- lift $ softmax x_score_p   -- [nS2,nS1,nN,nB]

  ------------------------- Eq (11)
  x_att    <- lift $ fBmm444_SS x_softmax xvpp  -- [hh,s1,n,b]
  x_att_p  <- lift $ transpose 1 2 x_att        -- [nHH,nN,nS,nB]
  x_att_pp <- lift $ merge x_att_p              -- [nH,nS,nB]

  ------------------------- Eq (13)
  x_out   <- lift $ einsum "h1,s,b|h2,h1|h2,s,b" x_att_pp wo
  x_out_p <- lift $ add x_out x             -- [nH,nS,nB]

  ------------------------- Eq (15)
  x_mlp1   <- lift $ einsum "h,s,b|4h,h|4h,s,b" x_out_p w1
  x_mlp1_p <- lift $ elementwise Relu [0,1,2] x_mlp1 -- [4*nH,nS,nB]

  ------------------------- Eq (17)
  x_mlp2   <- lift $ einsum "4h,s,b|h,4h|h,s,b" x_mlp1_p w2
  x_mlp2_p <- lift $ add x_mlp2 x_out_p           -- [nH,nS,nB]

  -- RECURSE TO DO THE REST OF THE LAYERS
  g_x_mlp2_p <- compute layers x_mlp2_p           -- [nH,nS,nB]

  -- BACKWARD COMPUTATION
  ------------------------- Eq (18)
  let g_x_mlp2    = g_x_mlp2_p -- [nH,nS,nB]
  let g_x_out_p2  = g_x_mlp2_p -- [nH,nS,nB]
  g_x_mlp1_p <- lift $ einsum "h,s,b|h,4h|4h,s,b" g_x_mlp2 w2
  g_w2       <- lift $ einsum "4h,s,b|h,s,b|h,4h" x_mlp1_p g_x_mlp2

  ------------------------- Eq (16)
  g_x_mlp1   <- lift $ elementwise Reluderiv [0,1,2] g_x_mlp1_p -- [4*nH, nS, nB]
  g_x_out_p1 <- lift $ einsum "fnH,nS,nB|fnH,nH|nH,nS,nB" g_x_mlp1 w1
  g_x_out_p  <- lift $ add g_x_out_p1 g_x_out_p2                -- [nH,nS,nB]
  g_w1       <- lift $ einsum "nH,nS,nB|fnH,nS,nB|fnH,nH" x_out_p g_x_mlp1

  ------------------------- Eq (14)
  let g_x_out = g_x_out_p -- [nH,nS,nB]
      g_x_4   = g_x_out_p -- [nH,nS,nB]
  g_x_att_pp <- lift $ einsum "nH1,nS,nB|nH1,nH2|nH2,nS,nB" g_x_out wo
  g_wo       <- lift $ einsum "h1,s,b|h2,s,b|h2,h1" x_att_pp g_x_out

  ------------------------- Eq (12)
  g_x_att_p   <- lift $ split nHH g_x_att_pp         -- [nHH,nN,nS,nB]
  g_x_att     <- lift $ transpose 1 2 g_x_att_p      -- [nHH,nS,nN,nB]
  g_x_softmax <- lift $ fBmm444_ST g_x_att xvpp      -- [s2,s1,n,b]
  g_x_v_pp    <- lift $ fBmm444_TS x_softmax g_x_att -- [hh,s2,n,b]

  ------------------------- Eq (10)
  crossd      <- lift $ fBmm444_TS x_softmax x_softmax -- [s,s,n,b]
  scrossd     <- lift $ elementwiseBinary Sub [0,1,2,3] [1,0,2,3] [0,1,2,3] x_softmax crossd
                                                       -- [s,s,n,b]
  g_x_score   <- lift $
    let alpha = (1.0 / (sqrt (fromIntegral nHH)))
     in fBmm444_SS_Alpha alpha g_x_softmax scrossd     -- [s,s,n,b]
  g_x_q_pp  <- lift $ fBmm444_ST g_x_score xkppp              -- [hh,s,n,b]
  g_x_k_ppp <- lift $ fBmm444_TT g_x_q_pp g_x_score           -- [hh,s,n,b]

  ------------------------- Eq ( 8)
  g_x_v_p <- lift $ transpose 1 2 g_x_v_pp
  g_x_v   <- lift $ merge g_x_v_p              -- [h,s,b]
  g_x_3   <- lift $ fBmm323_ST g_x_v wv        -- [h,s,b]
  g_wv    <- lift $ einsum "h1,s,b|h2,s,b|h2,h1" x g_x_v

  ------------------------- Eq ( 6)
  g_x_k_pp <- lift $ transpose 0 1 g_x_k_ppp
  g_x_k_p  <- lift $ transpose 1 2 g_x_k_pp
  g_x_k    <- lift $ merge g_x_k_p
  g_x_2    <- lift $ fBmm323_ST g_x_k wk
  g_wk     <- lift $ einsum "h1,s,b|h2,s,b|h2,h1" x g_x_k

  ------------------------- Eq ( 4)
  g_x_q_p <- lift $ transpose 1 2 g_x_q_pp
  g_x_q   <- lift $ merge g_x_q_p
  g_x_1   <- lift $ fBmm323_ST g_x_q wq
  g_wq    <- lift $ einsum "h1,s,b|h2,s,b|h2,h1" x g_x_q

  ------------------------- Eq ( 2)
  g_x_t0 <- lift $ add g_x_1 g_x_2
  g_x_t1 <- lift $ add g_x_3 g_x_4
  g_x    <- lift $ add g_x_t0 g_x_t1

  -- DO THE GRADIENT STEP
  descentParam thisLayer "wq" g_wq
  descentParam thisLayer "wv" g_wv
  descentParam thisLayer "wk" g_wk
  descentParam thisLayer "wo" g_wo
  descentParam thisLayer "w1" g_w1
  descentParam thisLayer "w2" g_w2

  -- RETURN UPDATED VALUE
  return g_x

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

fBmm323_SS = einsum "j,i,b|k,j|k,i,b"
fBmm323_TS = einsum "i,j,b|k,j|k,i,b"
fBmm323_ST = einsum "j,i,b|j,k|k,i,b"
fBmm323_TT = einsum "i,j,b|j,k|k,i,b"

fBmm444_SS_Alpha f = einsumAlpha "j,i,b2,b1|k,j,b2,b1|k,i,b2,b1" f
fBmm444_TS_Alpha f = einsumAlpha "i,j,b2,b1|k,j,b2,b1|k,i,b2,b1" f
fBmm444_ST_Alpha f = einsumAlpha "j,i,b2,b1|j,k,b2,b1|k,i,b2,b1" f
fBmm444_TT_Alpha f = einsumAlpha "i,j,b2,b1|j,k,b2,b1|k,i,b2,b1" f

fBmm444_SS = fBmm444_SS_Alpha 1.0
fBmm444_TS = fBmm444_TS_Alpha 1.0
fBmm444_ST = fBmm444_ST_Alpha 1.0
fBmm444_TT = fBmm444_TT_Alpha 1.0

