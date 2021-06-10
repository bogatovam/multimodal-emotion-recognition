from models.transformers.transformer import TransformerModel
import tensorflow as tf
import tensorflow_addons as tfa
import sys

from configs.dataset.modality import DatasetFeaturesSet
from dataset.manager.data_manager import DataManager
from dataset.preprocessor.multimodal_dataset_features_processor import MultimodalDatasetFeaturesProcessor
from trainers.simple_trainer import SimpleTrainer
from tensorboard.plugins.hparams import api as hp

sys.path.insert(0, './')

import configs.by_device_type.cpu_config as config

data_manager = DataManager(tf_record_path="D:/2021/hse/tfrecords",
                           batch_size=1)


def train_and_validate_model(board_path, cp_path, hyper_params: dict, config) -> dict:
    processor = MultimodalDatasetFeaturesProcessor(modalities_list=hyper_params['modality'])

    model = TransformerModel(
        regularizer=hyper_params['regularizer'](hyper_params['regularizer_lambda']) if hyper_params[
                                                                                           'regularizer'] is not None else None,
        activation=hyper_params['activation'],
        optimizer=hyper_params['optimizer'],
        num_layers=hyper_params['num_layers'],
        d_model=hyper_params['d_model'],
        num_heads=hyper_params['num_heads'],
        intermediate_fc_units_count=hyper_params['fc_units'],
        num_classes=7,
        max_features_count=10000,
        input_shape=hyper_params['modality'][0].config.input_shape,
        dropout_rate=hyper_params['dropout_rate'],
        weight_decay=0.000001,
        learning_rate=hyper_params['learning_rate'],
        cp_dir=cp_path,
        cp_name=config.CHECKPOINT_NAME,
        iter_per_epoch=config.NUM_ITER_PER_EPOCH
    )

    _, epoch = model.load()

    trainer = SimpleTrainer(
        dataset_processor=processor,
        model=model,
        data=data_manager,
        board_path=board_path,
        log_freq=config.LOG_AND_SAVE_FREQ_BATCH,
        num_epochs=1,
        initial_epoch=epoch,
        num_iter_per_epoch=config.NUM_ITER_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS,
        lr=config.LEARNING_RATE,
        create_dirs_flag=True
    )

    metrics = trainer.train()
    print(metrics)
    return metrics


def run_experiment(run_dir, cp_dir, hyper_params):
    board_dir = run_dir

    hparam_dir = run_dir + "/" + "hparam"

    with tf.summary.create_file_writer(hparam_dir).as_default():
        tmp_params = {}
        tmp_params.update(hyper_params)
        tmp_params['modality'] = hyper_params['modality'][0].name
        tmp_params['optimizer'] = hyper_params['optimizer'].__name__
        tmp_params['regularizer'] = hyper_params['regularizer'].__name__
        hp.hparams(tmp_params)
        metrics = train_and_validate_model(board_dir, cp_dir, hyper_params, config)
        for name, value in metrics.items():
            tf.summary.scalar(name, value[-1], step=1)


def main():
    with strategy.scope():
        session_num: int = 0
        initial_run = 0

        base_path = config.TENSORBOARD_DIR + "/" + "hp_tuning"
        cp_path = config.CHECKPOINT_DIR + "/" + "hp_tuning"

        modalities = [
            [DatasetFeaturesSet.AUDIO],
            [DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES],
            [DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES],
            [DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES],
            [DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES],
            [DatasetFeaturesSet.SHIMMERS],
            [DatasetFeaturesSet.SKELETON],
        ]

        regularizer_values = [tf.keras.regularizers.L2, tf.keras.regularizers.L1, None]
        regularizer_lambda_values = [0.01, 0.05, 0.1]
        activation_values = ['relu', 'tanh']
        optimizer_values = [tf.keras.optimizers.Adam, tfa.optimizers.AdamW]
        num_layers_values = [2, 4, 8]
        d_model_values = [256, 128, 512]
        num_heads_values = [8, 12, 16]
        fc_units_values = [512, 1024, 2048]
        dropout_rate_values = [0.15, 0.3, 0.5]
        learning_rate_values = [0.001, 0.0001, 0.00001]

        for learning_rate in learning_rate_values:
            for num_layers in num_layers_values:
                for num_heads in num_heads_values:
                    for fc_units in fc_units_values:
                        for d_model in d_model_values:
                            for optimizer in optimizer_values:
                                for activation in activation_values:
                                    for regularizer_lambda in regularizer_lambda_values:
                                        for regularizer in regularizer_values:
                                            for dropout_rate in dropout_rate_values:
                                                for modality in modalities:
                                                    if session_num < initial_run:
                                                        session_num += 1
                                                        continue

                                                    hyper_params: dict = {
                                                        'learning_rate': learning_rate,
                                                        'num_layers': num_layers,
                                                        'num_heads': num_heads,
                                                        'fc_units': fc_units,
                                                        'd_model': d_model,
                                                        'optimizer': optimizer,
                                                        'activation': activation,
                                                        'regularizer_lambda': regularizer_lambda,
                                                        'regularizer': regularizer,
                                                        'dropout_rate': dropout_rate,
                                                        'modality': modality,
                                                    }
                                                    run_name = "run-{}-M-{}-LR-{}-NL-{}-NH-{}-FC-{}-D-{}".format(
                                                        session_num,
                                                        modality[0].name,
                                                        learning_rate,
                                                        num_layers,
                                                        num_heads,
                                                        fc_units,
                                                        d_model)

                                                    print('--- Starting experiment: %s' % run_name)
                                                    print({h: hyper_params[h] for h in hyper_params})

                                                    exp_path = base_path + "/" + run_name
                                                    cp_exp_path = cp_path + run_name

                                                    run_experiment(exp_path, cp_exp_path, hyper_params)
                                                    session_num += 1


if __name__ == '__main__':
    main()
