"""Explicit common-rank phase-bank control; never change pooled subspaces."""


def constrain_phase_ranks(spaces,spec):
    if spec is None:return {}
    expected={f'{r}:{p}' for r,p in spaces if p>=0}
    if not isinstance(spec,dict) or set(spec)!=expected:
        raise ValueError('Common ranks must specify every supported phase bank exactly once')
    audit={}
    # Validate the entire map before mutating any basis.
    for (role,phase),space in spaces.items():
        if phase<0:continue
        key=f'{role}:{phase}';rank=spec[key]
        if type(rank) is not int or not 1<=rank<=space.rank:
            raise ValueError(f'Common rank for {key} must be a positive integer no larger than its FIT rank')
        audit[key]={'original_rank':space.rank,'common_rank':rank}
    for (role,phase),space in spaces.items():
        if phase>=0:
            space.rank=spec[f'{role}:{phase}'];space.basis=space.basis[:space.rank].copy()
    return audit
