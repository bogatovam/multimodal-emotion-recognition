import os
import sys

from dataset.manager.data_manager import DataManager
from dataset.preprocessor.face.feature_extractor import FeatureExtractor
from models.transformers.transformer import TransformerModel
from trainers.audio_extractor_trainer import SimpleTrainer

sys.path.insert(0, './')

import configs.by_device_type.cpu_config as config


def main():
    processor = FeatureExtractor(pretrained_model_path='../models/pretrained/rplus1',
                                 input_shape=(224, 224), frames_step=4,
                                 output_shape=(112, 112))

    data_manager = DataManager(dataset_processor=processor,
                               tf_record_path=os.path.join(config.DATASET_TF_RECORDS_PATH, config.NAME),
                               batch_size=config.BATCH_SIZE)

    model = TransformerModel(
        num_layers=2,
        d_model=512,
        num_heads=8,
        intermediate_fc_units_count=2048,
        num_classes=7,
        max_features_count=10000,
        input_shape=(39, 512),
        dropout_rate=0.1,
        weight_decay=0.00001,
        learning_rate=0.001,
        cp_dir=config.CHECKPOINT_DIR,
        cp_name=config.CHECKPOINT_NAME,
        iter_per_epoch=config.NUM_ITER_PER_EPOCH
    )

    _, epoch = model.load()

    trainer = SimpleTrainer(
        model=model,
        data=data_manager,
        board_path=config.TENSORBOARD_DIR,
        log_freq=config.LOG_AND_SAVE_FREQ_BATCH,
        num_epochs=config.NUM_EPOCHS,
        initial_epoch=epoch,
        num_iter_per_epoch=config.NUM_ITER_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS,
        lr=config.LEARNING_RATE,
        create_dirs_flag=True
    )

    metrics = trainer.train()
    print(metrics)


if __name__ == "__main__":
    main()
