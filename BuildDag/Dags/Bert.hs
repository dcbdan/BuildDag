module BuildDag.Dags.Bert (
  Params(..), bert,
) where

import Data.Map ( Map )
import qualified Data.Map as Map

import Data.IntSet ( IntSet )
import qualified Data.IntSet as IntSet

import BuildDag.Types
import BuildDag.Module
import BuildDag.Build

import qualified BuildDag.Graph as Graph

-- This is a copy of
--   https://github.com/codertimo/BERT-pytorch

data Params = Params {
    _nLayer    :: Int,
    _nQuery    :: Int,
    _nHead     :: Int,
    _nSequence :: Int,
    _fDropout  :: Float
  }

_nEmbed :: Params -> Int
_nEmbed params = (_nHead params) * (_nQuery params)

-- The input tensor contains a batch of sentences.
-- Each sentence contains words.
-- Each word has an embedding.
-- An embedding is actually the concatenation of the number of heads and queries.
--
-- The input tensor has shape (  Embed    , Sequence, Batch)
--                             ^^^^^^^^^^^
--                             Query, Head

bert :: Int -> Params -> (Map String Dims, BuildDagM ())
bert nBatch params =
  let mod = bertModule params
      sizings = Map.insert "X" [nEmbed, nSequence, nBatch] $
                  variablesMapping mod
      builtDag = do
        modBuilt <- initialize mod
        x <- initRandom "X" (-0.1) (0.1)
        forward11 modBuilt x
        return ()
      nEmbed    = _nEmbed params
      nSequence = _nSequence params
   in (sizings, builtDag)

bertModule :: Params -> Module ()
bertModule params = Module "bert" vars straightForward
  where
  nLayer = _nLayer params

  makeTransformerBlock which =
    transformerBlock ("transformerLayer"++show which) params

  vars = map makeTransformerBlock [1..nLayer]

transformerBlock :: String -> Params -> Module ()
transformerBlock name params =
  Module name vars straightForward
  where
  nEmbed   = _nEmbed params -- nQuery * nHead
  fDropout = _fDropout params

  attention   = multiHeadedAttention "mha" params
  feedForward = positionwiseFeedForward "pff" nEmbed (nEmbed * 4) fDropout

  vars = [
      sublayerConnection "sublayerIn"  attention   nEmbed fDropout,
      sublayerConnection "sublayerOut" feedForward nEmbed fDropout,
      dropoutModule fDropout
    ]

-- x -> [x,x,x] -> linearIn -> attention -> linearOut
-- esb         esb         qshb         qshb       esb
multiHeadedAttention :: String -> Params -> Module ()
multiHeadedAttention name params = Module name vars forward
  where
  vars = concat $ [
      map linearIn ["Q", "K", "V"],
      [attention (_fDropout params)],
      [linearOut "Out"]
    ]

  -- esb * eqh -> qshb                   e s b                     e q h    q s h b
  linearIn  nm = generalLinearModule nm [4,1,3]   (zip [nE,nQ,nH] [4,0,2]) [0,1,2,3]
  -- qshb * eqh -> esb                   q s h b                   e q h    e s b
  linearOut nm = generalLinearModule nm [3,1,4,2] (zip [nE,nQ,nH] [0,3,4]) [0,1,2]

  nQ = _nQuery params
  nH = _nHead params
  nE = nEmbed
  nEmbed = nQ*nH

  forward mods [x] = forward mods [x,x,x]
  forward [projQ, projK, projV, attention, projOut] [query, key, value] = do
    q <- forward11 projQ query
    k <- forward11 projK key
    v <- forward11 projV value
    x <- forward_1 attention [q, k, v]
    forward1_ projOut x

attention :: Float -> Module ()
attention dropoutParam = Module "attention" vars forward
  where
  vars = [softmaxModule, dropoutModule dropoutParam]
  forward [softmax, dropout] [q,k,v] = do
    -- Here, h,i are the broadcast dimensions

    -- q: kjih
    -- k: klih
    -- v: mlih
    -- output   mjih
    nK <- (toFloat . head) <$> getOutputDims q
    let scalar = 1.0 / (sqrt nK)
    -- kjih * klih -> ljih  |         k j i h   k l i h   l j i h
    scores <- contractionAlpha       [4,1,2,3] [4,0,2,3] [0,1,2,3] scalar q k
          -- ljih -> ljih
          >>= forward11 softmax
          -- ljih -> ljih
          >>= forward11 dropout
    -- ljih * mlih -> mjih  |  l j i h   m l i h   m j i h
    ret <- contraction        [4,1,2,3] [0,4,2,3] [0,1,2,3] scores v
    return [ret]

sublayerConnection :: String -> Module () -> Int -> Float -> Module ()
sublayerConnection name op size dropoutParam = Module name vars forward
  where
  vars = [
      normModule "norm" size (1e-6),
      op,
      dropoutModule dropoutParam
    ]
  forward modules [xIn] = do
    xOut <- head <$> straightForward modules [xIn]
    xConnected <- add xIn xOut
    return [xConnected]

normModule :: String -> Int -> Float -> Module ()
normModule name features eps = Module name normVars normForward
  where
  -- xIn, xOut is [features, j, i]
  --              [f,j,i]
  normVars = [
    Tensor "a_2" [features] (InitConstant 1.0) (),
    Tensor "b_2" [features] (InitConstant 0.0) ()]
  normForward [(Tensor _ _ _ a_2), (Tensor _ _ _ b_2)] [x] = do
    --  In ROW MAJOR pytorch:
    --    # [i,j,k] -> [i,j,1]
    --    mean = x.mean(-1, keepdim=True)
    --    # [i,j,k] -> [i,j,1]
    --    std = x.std(-1, keepdim=True)
    --    # [i,j,k]
    --    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    -- let m      = mean of x
    --     xm     = x - m
    --     s2     = 1/(features-1)*sum_out_f(xm .* xm)
    --     s      = sqrt(s2)
    --     sc     = s{i,j} + eps
    --     xma    = xm{i,j,k} *{i,j,k} a_2{k}
    --     xmas   = xma / sc
    --     xmasm  = xmas + b_2
    let n = (fromIntegral features :: Float)
    m  <- reductionAlpha CastableAdd [1,2] (1/n) x  -- ji
    xm <- elementwiseBinary Sub [0,1,2] [1,2] [0,1,2] x m          -- fji

    s2 <- contractionAlpha [0,1,2] [0,1,2] [1,2] (1/(n-1)) xm xm -- ji
    s  <- elementwise Sqrt [0,1] s2                              -- ji
    sc <- elementwise (AddScalar eps) [0,1] s                    -- ji

    xma    <- elementwiseBinary Mul [0,1,2] [0]   [0,1,2] xm   a_2  -- fji
    xmas   <- elementwiseBinary Div [0,1,2] [1,2] [0,1,2] xma  sc   -- fji
    xmasm  <- elementwiseBinary Add [0,1,2] [0]   [0,1,2] xmas b_2  -- fji

    return [xmasm]


dropoutModule :: Float -> Module ()
-- If p <= 0, the module is a no op...
-- Insert an identity module
dropoutModule p | p <= 0.0 = Module "" [] $ \_ xs -> return xs
-- Otherwise, actually insert a dropout node
dropoutModule p = Module "dropout" [] $ \_ [x] -> (:[]) <$> dropout p x


-- This is softmaxing the leading dimension, dimension 0.
softmaxModule :: Module ()
softmaxModule = Module "softmax" [] softmaxForward
  where
  -- softmax(x)_i = e^(x_i - c)/(\sum_i (e^(x_i - c)))
  -- where c = max_i x_i
  -- (the computation is the same for all c, but numerically more
  --  stable when the largest value of x is zero)
  softmaxForward [] [x] = do
    n <- length <$> getOutputDims x
    let allR = [0..(n-1)]
        lstR = [1..(n-1)]
    -- x                                                      ij
    c   <- reduction CastableMax lstR x                    -- j
    xc  <- elementwiseBinary Sub allR lstR allR x c        -- ij
    ex  <- elementwise Exp allR xc                         -- ij
    de  <- reduction CastableAdd  lstR ex                  -- j
    out <- elementwiseBinary Div allR lstR allR ex de      -- ij
    return [out]

positionwiseFeedForward :: String -> Int -> Int -> Float -> Module ()
positionwiseFeedForward name nEmbed nHidden dropoutParam =
  Module name vars forward
  where
  nE = nEmbed
  nH = nHidden
  -- xIn, xOut is eji
  vars = [
       -- eji,he->hji  |        e j i                 h e    h j i
      generalLinearModule "l1" [3,1,2] (zip [nH, nE] [0,3]) [0,1,2],
      --  hji,eh->eji  |        h j i                 e h    e j i
      generalLinearModule "l2" [3,1,2] (zip [nE, nH] [0,3]) [0,1,2],
      dropoutModule dropoutParam
    ]
  forward [l1, l2, dropout] [x0] =
    forward11 l1 x0             >>=
    -- TODO: support Gelu
    elementwise Relu [0,1,2]    >>=
    forward11 dropout           >>=
    forward1_ l2

toFloat :: Int -> Float
toFloat = fromIntegral
