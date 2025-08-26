from algorithm.pcsac import PCSAC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from env.StaticTask import StaticHandlingEnv
from robopal.commons.gym_wrapper import GymWrapper

TRAIN = 0
Method = "SAC"

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 51200 == 0:
            self.model.save(self.log_dir + f"/model_saved/policy_{self.n_calls}")
        return True


if TRAIN:
    # env = StaticHandlingEnv(render_mode='human')
    env = StaticHandlingEnv(render_mode=None)
else:
    env = StaticHandlingEnv(render_mode='human')
env = GymWrapper(env)

# Initialize the model
if Method == "SAC":
    log_dir = "log/PlaceTask/SAC"
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=log_dir,
    )

elif Method == "PCSAC":
    log_dir = "log/PlaceTask/PCSAC"
    model = PCSAC(
        # 'MlpPolicy',
        env,
        # verbose=1,
        tensorboard_log=log_dir,
    )

else:
    pass


if TRAIN:
    # train the model
    #model = CustomSAC.load(log_dir + f"/Final/SAC", env=env)
    model.learn(int(5e6), callback=TensorboardCallback(log_dir=log_dir))
    model.save(log_dir + f"/Final")

else:
    # Test the model
    model = SAC.load(log_dir + f"/model_saved/policy_460800")
    obs, info = env.reset()
    for i in range(int(1e6)):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        print(f"Step: {i}, Reward: {reward}, Info: {info}")
    env.close()
