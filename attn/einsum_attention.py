from torch import (einsum,
                   softmax,
                   concat,
                   unsqueeze,
                   Tensor)


def dotProductAttention(q, K, V) -> Tensor:
    """ Dot−Product Attention on one query .
    Args :
    q: a vector with shape [k]
    K: a matrix with shape [m, k]
    V: a matrix with shape [m, v]
    Returns :
    y : a vector with shape [v]
    """
    logits = einsum("k,mk−>m" , q , K)
    weights = softmax (logits)
    return einsum("m,mv−>v" , weights , V)

def multiheadAttention(x, M, P_q, P_k, P_v, P_o) -> Tensor:
    """ Multi−head Attention on one query .
    Args :
    x: a vector with shape [d]
    M : a matrix with shape [m, d]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [h, d, k]
    P_v: a tensor with shape [h, d, v]
    P_o: a tensor with shape [h, d, v]
    Returns :
    y : a vector with shape [d]
    """
    q = einsum (" d, hdk−>hk " , x, P_q)
    K = einsum ("md, hdk−>hmk" , M, P_k)
    V = einsum ("md, hdv−>hmv" , M, P_v)
    logits = einsum ("hk, hmk−>hm", q, K)
    weights = softmax (logits)
    o = einsum ("hm, hmv−>hv ", weights, V)
    y = einsum ("hv, hdv−>d ", o, P_o)
    return y

def multiheadAttentionBatched(
    X, M, mask , P_q, P_k, P_v, P_o ) -> Tensor:
    """ Multi−head Attention .
    Args :
    X: a tensor with shape [b, n, d]
    M: a tensor with shape [b, m, d]
    mask: a tensor with shape [b, h, n, m]
    P_q: a tensor with shape [ h, d, k]
    P_k: a tensor with shape [ h, d, k]
    P_v: a tensor with shape [ h, d, v]
    P_o: a tensor with shape [ h, d, v]
    Returns :
    Y: atensor with shape [b, n, d]
    """
    Q = einsum("bnd, hdk−>bhnk", X, P_q)
    K = einsum("bmd, hdk−>bhmk" , M, P_k)
    V = einsum("bmd, hdv−>bhmv" , M, P_v)
    logits = einsum("bhnk, bhmk−>bhnm", Q, K)
    weights = softmax(logits + mask)
    O = einsum ("bhnm, bhmv−>bhnv", weights, V)
    Y = einsum ("bhnv, hdv−>bnd", O, P_o)
    return Y

def multiheadSelfAttentionIncremental(
    x, prev_K, prev_V, P_q, P_k, P_v, P_o
    ) -> tuple[Tensor, Tensor, Tensor]:

    """ Multi−head Self−Attention (one step) .
    Args :
    x : a tensor with shape [b, d]
    prev_K : tensor with shape [b, h, m, k]
    prev_V : tensor with shape [b, h, m, v]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [h, d, k]
    P_v: a tensor with shape [h, d, v]
    P_o: a tensor with shape [h, d, v]
    Returns :
    y : a tensor with shape [b, d]
    new_K: tensor with shape [b, h, m+1, k]
    new_V: tensor with shape [b, h, m+1, v]
    """
    q = einsum("bd, hdk−>bhk ", x, P_q)
    new_K = concat(
        [prev_K, unsqueeze(einsum("bd, hdk−>bhk", M, P_k), axis=2)],
        axis=2)
    new_V = concat(
        [prev_V, unsqueeze(einsum("bd, hdv−>bhv" , M, P_v), axis=2)],
        axis=2)
    logits = einsum("bhk, bhmk−>bhm", q, new_K)
    weights = softmax(logits)
    O = einsum ("bhm, bhmv−>bhv", weights, new_V)
    y = einsum ("bhv, hdv−>bd", O, P_o)
    return y , new_K, new_V

def multiqueryAttentionBatched (
    X, M, mask, P_q, P_k, P_v, P_o ) -> Tensor:
    """ Multi−Query Attention .
    Args :
    X: a tensor with shape [b, n, d]
    M: a tensor with shape [b, m, d]
    mask : a tensor with shape [b, h, n, m]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [d, k]
    P_v: a tensor with shape [d, v]
    P_o: a tensor with shape [h, d, v]
    Returns :
    Y: a tensor with shape [b, n, d]
    """
    Q = einsum("bnd, hdk−>bhnk", X, P_q)
    K = einsum("bmd, dk−>bmk", M, P_k)
    V = einsum("bmd, dv−>bmv", M, P_v)
    logits = einsum("bhnk, bmk−>bhnm", Q, K)
    weights = softmax(logits + mask)
    O = einsum("bhnm, bmv−>bhnv", weights, V)
    Y = einsum("bhnv , hdv−>bnd", O, P_o)
    return Y

def multiquerySelfAttentionIncremental(
    x, prev_K, prev_V, P_q, P_k, P_v, P_o
    ) -> tuple[Tensor, Tensor, Tensor]:
    """ Multi−query Self−Attention (one step).
    Args :
    x : a tensor with shape [b, d]
    prev_K : tensor with shape [b, m, k]
    prev_V : tensor with shape [b, m, v]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [d, k]
    P_v: a tensor with shape [d, v]
    P_o: a tensor with shape [h, d, v]
    Returns :
    y : a tensor with shape [b, d]
    new_K: tensor with shape [b, m+1, k]
    new_V: tensor with shape [b, m+1, v]
    """
    q = einsum("bd, hdk−>bhk", x , P_q)
    K = concat(
        [ prev_K, unsqueeze(einsum("bd, dk−>bk", M, P_k), axis=2)],
        axis=2)
    V = concat(
        [prev_V , unsqueeze(einsum("bd, dv−>bv", M, P_v), axis=2)],
        axis=2)
    logits = einsum("bhk, bmk−>bhm", q, K)
    weights = softmax (logits)
    O = einsum("bhm, bmv−>bhv", weights, V)
    y = einsum("bhv, hdv−>bd", O, P_o)
    return y , K, V
