import os
import argparse
from env import make_env
from algorithm import SAC, PCSAC, TensorboardCallback


def main():
    parser = argparse.ArgumentParser(description="Running an experiment.")
    parser.add_argument("--train", type=int, default=1, help="Whether to train the model (default: True).")
    parser.add_argument("--method", type=str, default="SAC", help="Choose an algorithm (default: SAC).")
    parser.add_argument("--task", type=str, default="Reaching", help="Choose a task (default: Reaching).")
    parser.add_argument("--render-mode", type=str, default=None, help="The render mode (default: None)")
    parser.add_argument("--iterations", type=int, default=2e5, help="The total training iterations.")
    parser.add_argument("--test-iterations", type=int, default=1e6, help="The iterations for testing.")
    args = parser.parse_args()

    # create environments
    render_mode = args.render_mode if args.train else 'human'
    env = make_env(args.task, render_mode=render_mode)

    log_dir = f"log/{args.task}/{args.method}"
    if args.method == "SAC":
        model = SAC(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=log_dir)
    elif args.method == "PCSAC":
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
    elif args.method == "MTSAC":
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
    else:
        raise NotImplementedError

    if args.train:
        callback = TensorboardCallback(log_dir=log_dir)
        model.learn(int(args.iterations), callback=callback)
        model.save(os.path.join(log_dir, "Final"))
    else:
        # Test the model
        model = SAC.load(os.path.join(log_dir, f"Final"), env=env)  # f"model_saved/policy_51200"
        obs, info = env.reset()
        for i in range(int(args.test_iterations)):
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
            print(f"Step: {i}, Reward: {reward}, Info: {info}")
        env.close()


if __name__ == "__main__":
    main()
