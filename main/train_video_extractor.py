import os
import sys

from dataset.manager.data_manager import DataManager
from dataset.preprocessor.face.ft_video_modality_preprocessor import VideoModalityPreprocessor, VideoFeatureExtractor
from models.fine_tune_model import FineTuneModel
from trainers.audio_extractor_trainer import SimpleTrainer

sys.path.insert(0, './')

import configs.ramas_default_config as config


def main():
    processor = VideoModalityPreprocessor(extractor=VideoFeatureExtractor.C3D)

    data_manager = DataManager(dataset_processor=processor,
                               dataset_size=config.DATASET_SIZE,
                               tf_record_path=os.path.join(config.DATASET_TF_RECORDS_PATH, config.NAME),
                               batch_size=config.BATCH_SIZE)

    model = FineTuneModel(
        extractor=VideoFeatureExtractor.C3D,
        pretrained_model_path='../models/pretrained/c3d.h5',
        cp_dir=config.CHECKPOINT_DIR,
        cp_name=config.CHECKPOINT_NAME,
        learning_rate=config.LEARNING_RATE,
        iter_per_epoch=config.NUM_ITER_PER_EPOCH
    )

    model.load()

    trainer = SimpleTrainer(
        model=model,
        data=data_manager,
        board_path=config.TENSORBOARD_DIR,
        log_freq=config.LOG_AND_SAVE_FREQ_BATCH,
        num_epochs=config.NUM_EPOCHS,
        num_iter_per_epoch=config.NUM_ITER_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS,
        lr=config.LEARNING_RATE,
        create_dirs_flag=True
    )

    metrics = trainer.train()
    print(metrics)


if __name__ == "__main__":
    main()
