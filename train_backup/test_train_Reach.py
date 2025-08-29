from stable_baselines3.common.callbacks import BaseCallback
from env.ReachTask import ReachHandlingEnv
from robopal.commons.gym_wrapper import GymWrapper
from algorithm.kcrl import PCSAC
import os

TRAIN = 1


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        # 创建模型保存目录
        model_save_dir = os.path.join(self.log_dir, "model_saved")
        os.makedirs(model_save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % 51200 == 0:
            self.model.save(os.path.join(self.log_dir, f"model_saved/policy_{self.n_calls}"))
        return True


log_dir = "../log/ReachTask/SAC"

# 创建日志目录
os.makedirs(log_dir, exist_ok=True)

if TRAIN:
    env = ReachHandlingEnv(render_mode=None)
else:
    env = ReachHandlingEnv(render_mode=None)
env = GymWrapper(env)

# Initialize the model
model = PCSAC(
    env=env,  # 只传入环境对象
    num_tasks=3,  # 根据实际情况调整
    num_param_sets=5,  # 根据实际情况调整
    learning_rate=3e-4,  # 根据实际情况调整
    tau=0.005,  # 根据实际情况调整
    gamma=0.99,  # 根据实际情况调整
    use_sde=False,  # 根据实际情况调整
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
