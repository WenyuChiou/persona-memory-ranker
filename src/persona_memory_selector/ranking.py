"""Gold-blind selection on one common candidate set."""
from __future__ import annotations
import numpy as np

HEADER = ('Behavioral exemplars below are reference examples, not your own experiences. '
          'Do not claim you lived these events. Owned episodes are identified separately.\n')


def validate(rows):
    ids, sources = set(), set()
    for r in rows:
        for key in ('memory_id','text','source_ref'):
            if not isinstance(r.get(key),str) or not r[key].strip():
                raise ValueError(f'Missing {key}')
        if r['memory_id'] in ids or r['source_ref'] in sources:
            raise ValueError('Duplicate memory ID or source')
        ids.add(r['memory_id']); sources.add(r['source_ref'])
        if r.get('memory_type') not in ('behavioral_exemplar','owned_episode'):
            raise ValueError('Unknown memory_type')
        if r['memory_type'] == 'owned_episode' and not r.get('owner_id'):
            raise ValueError('Owned episode requires owner_id')
        for k in ('similarity','trait_score'):
            if not np.isfinite(r.get(k, np.nan)):
                raise ValueError(f'Invalid {k}')
        if not 0 <= r['trait_score'] <= 1:
            raise ValueError('Invalid trait probability')


def rank_candidates(rows, method, weight=.5, shuffle=False):
    validate(rows)
    if method not in ('A','B','C','D') or not 0 <= weight <= 1:
        raise ValueError('Invalid method or weight')
    if not rows or method == 'A':
        return []
    sim = np.clip(np.array([r['similarity'] for r in rows]), 0, 1)
    prob = np.array([r['trait_score'] for r in rows])
    if method == 'B':
        scores = sim
    elif method == 'C':
        scores = (1-weight)*sim + weight*prob
    else:
        # Bipartite example/cluster edges and one requested-trait node.
        # The same 50 candidates are retained; graph cannot add hidden evidence.
        cluster = [r['cluster_id'] for r in rows]
        if shuffle:
            cluster = np.random.default_rng(310).permutation(cluster).tolist()
        unique = sorted(set(cluster))
        n, size = len(rows), len(rows)+len(unique)+1
        adj = np.zeros((size,size))
        for i, c in enumerate(cluster):
            j = n+unique.index(c)
            adj[i,j] = adj[j,i] = 1
            adj[i,-1] = adj[-1,i] = prob[i]
        sums = adj.sum(axis=1)
        transition = np.divide(adj,sums[:,None],out=np.zeros_like(adj),where=sums[:,None]>0)
        seed = np.zeros(size)
        seed[:n] = (1-weight)*(sim/sim.sum() if sim.sum() else np.ones(n)/n)
        seed[-1] = weight
        p = seed.copy()
        for _ in range(500):
            q = .15*seed + .85*(p @ transition + p[sums==0].sum()*seed)
            if np.abs(q-p).sum() < 1e-12:
                p=q; break
            p=q
        else:
            raise RuntimeError('PageRank did not converge')
        scores = p[:n]
    return sorted([dict(r, rank_score=float(s)) for r,s in zip(rows,scores)],
                  key=lambda r: (-r['rank_score'],r['memory_id']))


def memory_block(rows):
    if not rows:
        return ''
    return HEADER + '\n\n'.join(f"[{r['memory_type']} {r['memory_id']} {r['source_ref']}]\n{r['text']}" for r in rows)


def pack(rows, count, budget=2000, limit=5):
    if budget < 0 or limit < 0:
        raise ValueError('Negative context budget')
    selected=[]
    for r in rows:
        if len(selected) >= limit:
            break
        if count(memory_block(selected+[r])) <= budget:
            selected.append(r)
    return selected
