import os
from stable_baselines3.common.callbacks import BaseCallback


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
