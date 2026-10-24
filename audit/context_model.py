"""Benchmark-context shrinkage, using ONLY evidence supplied in the current call.

The optional peer correction falls back exactly to Beta(1.5, 1.5) when no
labels from different visible subjects on the same benchmark are supplied.
No cross-call state, network, training at runtime, hidden IDs, or files.
"""
PRIOR_STRENGTH=3.0
PEER_MIX=0.5  # research default; final exported value must follow inner validation

def predict(input: list[dict],labeled: list|None=None)->float:
    subject,item=input
    benchmark=item.get('benchmark_id') if isinstance(item,dict) else None
    if not isinstance(benchmark,str) or not benchmark:return .5
    own_k=own_n=peer_k=peer_n=0
    for entry in labeled or ():
        try:(other,observed_item),y=entry
        except (TypeError,ValueError):continue
        if not isinstance(observed_item,dict) or observed_item.get('benchmark_id')!=benchmark:continue
        if y not in (0,1,0.,1.):continue
        if other==subject:own_k+=int(y);own_n+=1
        else:peer_k+=int(y);peer_n+=1
    peer_mean=(1.5+peer_k)/(3.+peer_n)
    prior=.5+PEER_MIX*(peer_mean-.5)
    return float((own_k+PRIOR_STRENGTH*prior)/(own_n+PRIOR_STRENGTH))
