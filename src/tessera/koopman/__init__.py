"""tessera.koopman — explicit-latent Koopman with time-delay embedding.

Implements the closed-form latent Koopman per docs/shipped/koopman.md:

    Encoder    E : pd → k         (past-stack to latent, SVD of prediction operator)
    Koopman    K : k → k          (linear dynamics on latent)
    Decoder    D : k → d          (latent to next observation)

    ŷ_{t+h}  =  D K^h E ỹ_t       (single matrix multiply per horizon)

where ỹ_t is the p-lag past-stack of observation history.

Distinct from N4SID by having SEPARATE encoder E and decoder D — N4SID ties
them via shared C. Distinct from EDMD by reducing rank to k SMALLER than pd
in observable space, forcing a bottleneck.

Public API:
    LatentKoopman           — the model class
"""
from .model import LatentKoopman

__all__ = ["LatentKoopman"]
