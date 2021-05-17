from base.base_train import BaseTrain
from dataset.manager.data_manager import DataManager
from models.audio.audio_extractor_model import FineTuneModel
from utils.logger import Logger


class FineTuneTrainer(BaseTrain):

    def __init__(self,
                 model: FineTuneModel,
                 data: DataManager,
                 board_path: str,
                 log_freq: int,
                 lr: float,
                 num_epochs: int,
                 num_iter_per_epoch,
                 validation_steps,
                 create_dirs_flag):
        self._learning_rate = lr
        self._board_path = board_path
        self._log_freq = log_freq
        self._num_epochs = num_epochs
        self._num_iter_per_epoch = num_iter_per_epoch
        self._validation_steps = validation_steps
        self._create_dirs_flag = create_dirs_flag
        super(FineTuneTrainer, self).__init__(model, data)

    def train(self):
        training_dataset = self.data.build_training_dataset()
        validation_dataset = self.data.build_validation_dataset()

        train_model = self.model.get_train_model()

        callbacks = [*self.model.get_model_callbacks(),
                     self._get_terminate_on_nan_callback()]

        history = train_model.fit(
            training_dataset,
            epochs=self._num_epochs,
            steps_per_epoch=self._num_iter_per_epoch,
            validation_steps=self._validation_steps,
            validation_data=validation_dataset,
            callbacks=callbacks)

        return history.history
