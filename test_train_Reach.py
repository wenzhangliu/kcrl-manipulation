from stable_baselines3.common.callbacks import BaseCallback
from env.ReachTask import ReachHandlingEnv
from robopal.commons.gym_wrapper import GymWrapper
from algorithm.pcsac import PCSAC
from stable_baselines3 import SAC
import os
import argparse


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


def main():
    parser = argparse.ArgumentParser(description="Running an experiment.")
    parser.add_argument("--train", type=int, default=1, help="Whether to train the model (default: True).")
    parser.add_argument("--method", type=str, default="SAC", help="Choose an algorithm (default: SAC).")
    parser.add_argument("--render-mode", type=str, default='human', help="The render mode (default: None)")
    parser.add_argument("--iterations", type=int, default=1e6, help="The render mode (default: None)")
    args = parser.parse_args()

    # create environments
    render_mode = args.render_mode
    env = ReachHandlingEnv(render_mode=render_mode)
    env = GymWrapper(env)

    log_dir = f"log/ReachTask/{args.method}"
    if args.method == "SAC":
        model = SAC(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=log_dir)
    elif args.method == "PCSAC":
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
    else:
        raise NotImplementedError

    if args.train:
        callback = TensorboardCallback(log_dir=log_dir)
        model.learn(int(args.iterations), callback=callback)
        model.save(os.path.join(log_dir, "Final"))
    else:
        # Test the model
        model = SAC.load(os.path.join(log_dir, f"model_saved/policy_51200"), env=env)
        obs, info = env.reset()
        for i in range(int(args.iterations)):
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
            print(f"Step: {i}, Reward: {reward}, Info: {info}")
        env.close()


if __name__ == "__main__":
    main()
