import os
import numpy as np
from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ENV_IDS = []

for task in ["Reach", "Slide", "Push", "PickAndPlace", "Stack", "Flip"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            env_id = f"Panda{task}{control_suffix}{reward_suffix}-v3"

            register(
                id=env_id,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=100 if task == "Stack" else 50,
            )

            ENV_IDS.append(env_id)

object_size = 0.04
register(
    id=f"PandaStackSimple-g-01x01-o1-02x02-o2-025x025-v3",
    entry_point=f"panda_gym.envs:PandaStackSimpleEnv",
    kwargs={"reward_type": "dense", "control_type": "joints", "goal1": np.array([0.1, 0.1, object_size / 2]),
                                                                      "goal2": np.array([0.1, 0.1, 3 * object_size / 2]),
                                                                      "obj1": np.array([0.2, 0.2, object_size / 2]),
                                                                      "obj2": np.array([0.25, 0.25, 3 * object_size / 2])},
    max_episode_steps=100
)

register(
    id=f"PandaPushSimple-g-m01xm01-o-01x01-v3",
    entry_point=f"panda_gym.envs:PandaPushSimpleEnv",
    kwargs={"reward_type": "dense", "control_type": "joints", "goal": np.array([-0.05, 0, object_size / 2]),
                                                              "obj": np.array([0.05, 0, object_size / 2])},
    max_episode_steps=500
)
