import os
from algorithm.pcsac import PCSAC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from env.GraspTask import GraspHandlingEnv
from robopal.commons.gym_wrapper import GymWrapper

TRAIN = 1
Method = "PCSAC"

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        model_save_dir = os.path.join(self.log_dir, "model_saved")
        os.makedirs(model_save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % 51200 == 0:
            self.model.save(os.path.join(self.log_dir, f"model_saved/policy_{self.n_calls}"))
        return True

if TRAIN:
    env = GraspHandlingEnv(render_mode=None)
else:
    env = GraspHandlingEnv(render_mode=None)
env = GymWrapper(env)

# Initialize the model
if Method == "SAC":
    log_dir = "log/GraspTask/SAC"
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=log_dir,
    )

elif Method == "PCSAC":
    log_dir = "log/GraspTask/KCRL"
    model = PCSAC(
        env=env,
        num_tasks=2,
        num_param_sets=5,
        learning_rate=3e-4,
        tau=0.005,
        gamma=0.99,
        use_sde=False,
        tensorboard_log=log_dir,
    )

elif Method == "MTSAC":
    log_dir = "log/GraspTask/MTSAC"
    model = PCSAC(
        env=env,
        num_tasks=3,
        num_param_sets=5,
        learning_rate=3e-4,
        tau=0.005,
        gamma=0.99,
        use_sde=False,
        tensorboard_log=log_dir,
    )


if TRAIN:
    # Train the model
    callback = TensorboardCallback(log_dir=log_dir)
    model.learn(int(1e6), callback=callback)
    model.save(os.path.join(log_dir, "Final"))
else:
    # Test the model
    model = PCSAC.load(os.path.join(log_dir, f"model_saved/policy_204800"), env=env)
    obs, info = env.reset()
    for i in range(int(1e6)):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        print(f"Step: {i}, Reward: {reward}, Info: {info}")
    env.close()
    env.close()
