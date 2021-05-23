import sys
import tensorflow as tf

from configs.dataset.modality import DatasetFeaturesSet
from dataset.manager.data_manager import DataManager
from dataset.preprocessor.multimodal_dataset_features_processor import MultimodalDatasetFeaturesProcessor
from models.transformers.transformer import TransformerModel
from trainers.audio_extractor_trainer import SimpleTrainer
from tensorboard.plugins.hparams import api as hp

sys.path.insert(0, './')

import configs.by_device_type.cpu_config as config


def train_and_validate_model(board_path, cp_path, hyper_params: dict, config) -> dict:
    processor = MultimodalDatasetFeaturesProcessor(modalities_list=hyper_params['modality'])

    data_manager = DataManager(dataset_processor=processor,
                               tf_record_path="gs://ramas_tpu/ramas_vas",
                               batch_size=config.BATCH_SIZE)

    model = TransformerModel(
        num_layers=hyper_params['num_layers'],
        d_model=512,
        num_heads=hyper_params['num_heads'],
        intermediate_fc_units_count=2048,
        num_classes=7,
        max_features_count=10000,
        input_shape=hyper_params['modality'][0].config.input_shape,
        dropout_rate=hyper_params['dropout_rate'],
        weight_decay=0.00001,
        learning_rate=hyper_params['learning_rate'],
        cp_dir=cp_path,
        cp_name=config.CHECKPOINT_NAME,
        iter_per_epoch=config.NUM_ITER_PER_EPOCH
    )

    _, epoch = model.load()

    trainer = SimpleTrainer(
        model=model,
        data=data_manager,
        board_path=board_path,
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
    return metrics


def run_experiment(run_dir, cp_dir, hyper_params):
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])

    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        board_dir = run_dir

        hparam_dir = run_dir + "/" + "hparam"

        with tf.summary.create_file_writer(hparam_dir).as_default():
            tmp_params = {}
            tmp_params.update(hyper_params)
            tmp_params['modality'] = hyper_params['modality'][0].name
            hp.hparams(tmp_params)
            metrics = train_and_validate_model(board_dir, cp_dir, hyper_params, config)
            for name, value in metrics.items():
                tf.summary.scalar(name, value[-1], step=1)


def main():
    session_num: int = 0
    initial_run = 1

    base_path = config.TENSORBOARD_DIR + "/" + "hp_tuning"
    cp_path = config.CHECKPOINT_DIR + "/" + "hp_tuning"

    modalities = [
        [DatasetFeaturesSet.AUDIO],
        [DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES],
        [DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES],
        [DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES],
        [DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES],
        [DatasetFeaturesSet.VIDEO_FACE_IR50_FEATURES],
        # [DatasetFeaturesSet.SKELETON],
    ]
    dropout_rate_values = [0.1, 0.2, 0.3]
    num_layers_values = [2, 4, 8]
    num_heads_values = [8, 12, 16]
    num_heads_values = [8, 12, 16]
    learning_rate_exp_values = [-2, -3, -4]
    for dropout_rate in dropout_rate_values:
        for num_layers in num_layers_values:
            for num_heads in num_heads_values:
                for learning_rate_exp in learning_rate_exp_values:
                    for modality in modalities:
                        if session_num < initial_run:
                            session_num += 1
                            continue

                        learning_rate = pow(10, learning_rate_exp)
                        hyper_params: dict = {
                            'learning_rate': learning_rate,
                            'dropout_rate': dropout_rate,
                            'num_layers': num_layers,
                            'num_heads': num_heads,
                            'modality': modality,
                        }
                        run_name = "run-{}-M-{}-LR-{}-SR-{}-NL-{}-NH-{}".format(session_num,
                                                                                modality[0].name,
                                                                                learning_rate_exp,
                                                                                dropout_rate,
                                                                                num_layers,
                                                                                num_heads)
                        print('--- Starting experiment: %s' % run_name)
                        print({h: hyper_params[h] for h in hyper_params})

                        exp_path = base_path + "/" + run_name
                        cp_exp_path = cp_path + run_name

                        run_experiment(exp_path, cp_exp_path, hyper_params)
                        session_num += 1


if __name__ == '__main__':
    main()
