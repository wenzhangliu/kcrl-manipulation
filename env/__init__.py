from .ConveyorTask import ConveyorHandlingEnv1
from .GraspTask import GraspHandlingEnv
from .ReachTask import ReachHandlingEnv
from .StaticTask import StaticHandlingEnv
from robopal.commons.gym_wrapper import GymWrapper


RegistryEnvs = {
    "Reaching": ReachHandlingEnv,
    "Grasping": GraspHandlingEnv,
    "Placing": StaticHandlingEnv,
    "MovingConveyor": ConveyorHandlingEnv1,
}


def make_env(env_name, **kwargs):
    env = RegistryEnvs[env_name](**kwargs)
    env = GymWrapper(env)
    return env
