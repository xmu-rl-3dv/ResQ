from functools import partial
import sys
import os

from smac.env import MultiAgentEnv, StarCraft2Env
try:
    from .particle import Particle
except:
    pass
from .stag_hunt import StagHunt

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
try:
    REGISTRY["particle"] = partial(env_fn, env=Particle)
except:
    pass
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
