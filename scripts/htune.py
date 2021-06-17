from datetime import datetime

from configs.dataset.modality import DatasetFeaturesSet
from models.transformers.transformer import CustomSchedule


def train_and_validate_model(board_path, cp_path, hyper_params: dict, config) -> dict:
    processor = MultimodalDatasetFeaturesProcessor(modalities_list=hyper_params['modality'])

    additional_args = {}
    model = MultiModelTransformerModel(
        fusion_type=hyper_params['fusion'],
        co_attention=hyper_params['coattention'],
        modalities_list=hyper_params['modality'],
        optimizer=hyper_params['optimizer'],
        pooling_size=4,
        num_layers=hyper_params['num_layers'],
        d_model=hyper_params['dmodel'],
        num_heads=hyper_params['num_heads'],
        intermediate_fc_units_count=1024,
        num_classes=7,
        max_features_count=10000,
        dropout_rate=hyper_params['dropout_rate'],
        weight_decay=0.00001,
        learning_rate=hyper_params['learning_rate'],
        cp_dir=cp_path,
        cp_name=CHECKPOINT_NAME
    )

    _, epoch = model.load()

    trainer = SimpleTrainer(
        dataset_processor=processor,
        model=model,
        data=data_manager,
        board_path=board_path,
        log_freq=LOG_AND_SAVE_FREQ_BATCH,
        num_epochs=20,
        initial_epoch=epoch,
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
        tmp_params['modality'] = ""
        for m in hyper_params['modality']:
            tmp_params['modality'] += m.name
        tmp_params['optimizer'] = hyper_params['optimizer'].__name__
        tmp_params['learning_rate'] = hyper_params['learning_rate'].__class__.__name__
        hp.hparams(tmp_params)
        metrics = train_and_validate_model(board_dir, cp_dir, hyper_params, config)
        for name, value in metrics.items():
            tf.summary.scalar(name, value[-1], step=1)


def main():
    with strategy.scope():
        session_num: int = 0
        initial_run = 37

        base_path = TENSORBOARD_DIR + "/" + "hp_tuning"
        cp_path = CHECKPOINT_DIR + "/" + "hp_tuning"

        fusion_type_values = ['concatenation', 'sum', 'fbp', 'mha']
        co_attention_values = [True, False]
        modalities = [
            DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES,
            DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES,
            DatasetFeaturesSet.SKELETON,
            DatasetFeaturesSet.SHIMMERS,
            DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES,
            DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES,
        ]

        c1 = {
            "num_layers": 6,
            "num_heads": 4,
            "dmodel": 512
        }
        c2 = {
            "num_layers": 2,
            "num_heads": 8,
            "dmodel": 256
        }
        configs = [c1, c2]

        scheduler_values = [
            tf.keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=300, decay_rate=0.2, staircase=True),
            CustomSchedule(512)]
        optimizers = [tf.keras.optimizers.Adam, tfa.optimizers.AdamW]
        for c in configs:
            for optimizer in optimizers:
                for scheduler in scheduler_values:
                    for fusion in fusion_type_values:
                        for co_attention in co_attention_values:
                            modalities_exp = [DatasetFeaturesSet.AUDIO]
                            for modality in modalities:
                                modalities_exp.append(modality)
                                if modality == DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES:
                                    del modalities_exp[2]
                                if session_num < initial_run or fusion == 'mha' or fusion == 'sum':
                                    session_num += 1
                                    continue
                                hyper_params: dict = {
                                    'dmodel': c['dmodel'],
                                    'learning_rate': scheduler,
                                    'dropout_rate': 0.15,
                                    'num_layers': c['num_layers'],
                                    'num_heads': c['num_heads'],
                                    'modality': modalities_exp,
                                    'coattention': co_attention,
                                    'optimizer': optimizer,
                                    'fusion': fusion,
                                }
                                run_name = "run-{}-M-{}-fusion-{}-CA-{}".format(session_num, modality.name, fusion,
                                                                                co_attention)
                                print('--- Starting experiment: %s' % run_name)
                                print({h: hyper_params[h] for h in hyper_params})

                                exp_path = base_path + "/" + modality.name + "/" + run_name
                                cp_exp_path = cp_path + modality.name + "/" + run_name

                                run_experiment(exp_path, cp_exp_path, hyper_params)
                                session_num += 1


if __name__ == '__main__':
    main()
