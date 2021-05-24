import sys

from configs.dataset.modality import DatasetFeaturesSet
from dataset.manager.data_manager import DataManager
from dataset.preprocessor.multimodal_dataset_features_processor import MultimodalDatasetFeaturesProcessor
from models.transformers.multi_modal_transformer import MultiModelTransformerModel
from models.transformers.transformer import TransformerModel
from trainers.audio_extractor_trainer import SimpleTrainer
from configs.dataset.modality import DatasetFeaturesSet
from dataset.manager.data_manager import DataManager
from dataset.preprocessor.multimodal_dataset_features_processor import MultimodalDatasetFeaturesProcessor
from trainers.audio_extractor_trainer import SimpleTrainer
from tensorboard.plugins.hparams import api as hp
sys.path.insert(0, './')
import tensorflow as tf
import configs.by_device_type.cpu_config as config

from models.transformers.multi_modal_transformer import MultiModelTransformerModel


data_manager = DataManager(tf_record_path="D:/2021/hse/tfrecords/emotion-transformer",
                               batch_size=config.BATCH_SIZE)

def train_and_validate_model(board_path, cp_path, hyper_params: dict, config) -> dict:
   processor = MultimodalDatasetFeaturesProcessor(modalities_list=hyper_params['modality'])

   model = MultiModelTransformerModel(
       fusion_type=hyper_params['fusion'],
       co_attention=hyper_params['coattention'],
       modalities_list=hyper_params['modality'],
       pooling_size=4,
       num_layers=hyper_params['num_layers'],
       d_model=hyper_params['dmodel'],
       num_heads=hyper_params['num_heads'],
       intermediate_fc_units_count=2048,
       num_classes=7,
       max_features_count=10000,
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
       num_epochs=20,
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

        hparam_dir = run_dir +"/" + "hparam"

        with tf.summary.create_file_writer(hparam_dir).as_default():
            tmp_params = {}
            tmp_params.update(hyper_params)
            tmp_params['modality'] = ""
            for m in hyper_params['modality']:
              tmp_params['modality'] += m.name
            hp.hparams(tmp_params)
            metrics = train_and_validate_model(board_dir, cp_dir, hyper_params, config)
            for name, value in metrics.items():
                tf.summary.scalar(name, value[-1], step=1)


def main():
    # with strategy.scope():
        session_num: int = 0
        initial_run = 0

        base_path = config.TENSORBOARD_DIR + "/" + "hp_tuning"
        cp_path =  config.CHECKPOINT_DIR + "/" + "hp_tuning"

        fusion_type_values = ['sum','concatenation','fbp','mha']
        co_attention_values = [True, False]
        modalities = [
                      DatasetFeaturesSet.VIDEO_FACE_VGG_FEATURES,
                      DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES,
                      DatasetFeaturesSet.VIDEO_FACE_IR50_FEATURES,
                      DatasetFeaturesSet.VIDEO_SCENE_IV3_FEATURES,
                      # DatasetFeaturesSet.VIDEO_FACE_R2PLUS1_FEATURES,
                      # [DatasetFeaturesSet.SKELETON],
        ]
        c1 =  {
          "num_layers": 2,
          "num_heads": 4,
          "dropout_rate": 0.3,
        }
        c2 =  {
            "num_layers": 2,
            "num_heads": 8,
            "dropout_rate": 0.1,
        }
        configs = [c1, c2]
        for c in configs:
            for fusion in fusion_type_values:
                for co_attention in co_attention_values:
                    modalities_exp = [DatasetFeaturesSet.AUDIO]
                    for modality in modalities:
                            if session_num < initial_run:
                                session_num += 1
                                continue
                            modalities_exp.append(modality)
                            hyper_params: dict = {
                                'dmodel': 256,
                                'learning_rate': 0.001,
                                'dropout_rate': c['dropout_rate'],
                                'num_layers': c['num_layers'],
                                'num_heads': c['num_heads'],
                                'modality': modalities_exp,
                                'coattention': co_attention,
                                'fusion': fusion,
                            }
                            run_name = "run-{}-M-{}".format(session_num, modality.name)
                            print('--- Starting experiment: %s' % run_name)
                            print({h: hyper_params[h] for h in hyper_params})

                            exp_path = base_path + "/" + run_name
                            cp_exp_path =cp_path +  run_name

                            run_experiment(exp_path, cp_exp_path, hyper_params)
                            session_num += 1


if __name__ == '__main__':
    main()
