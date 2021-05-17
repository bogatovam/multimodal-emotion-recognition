import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeature
from dataset.preprocessor.feature_extractors_metadata import VideoFeatureExtractor


class FaceModalityPreprocessor(BaseDatasetProcessor):

    def __init__(self, extractor: VideoFeatureExtractor, input_face_size, frames_step, pretrained_model_path):
        self._extractor = extractor
        self._output_shape = extractor.output_shape[0]
        self._frames_step = frames_step
        self._input_face_size = input_face_size

        self._pretrained_feature_extractor = \
            tf.keras.models.load_model(pretrained_model_path, compile=False).signatures["serving_default"]
        self._pretrained_feature_extractor.trainable = False

        self._feature_description = {
            DatasetFeature.L3.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.VIDEO_FACE_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.CLASS.name: tf.io.FixedLenFeature([], tf.string)
        }

    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        dataset = dataset.map(self.map_record_to_dictionary_of_tensors, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self._C3D_extract_frames_features, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self.concat_with_labels, num_parallel_calls=parallel_calls)
        return dataset

    @tf.function
    def map_record_to_dictionary_of_tensors(self, serialized_example: tf.train.Example) -> dict:
        return self._decode_example(serialized_example)

    @tf.function
    def concat_with_labels(self, example: tf.train.Example):

        clazz = example[DatasetFeature.CLASS.name]
        clazz = tf.expand_dims(clazz, 0)
        clazz = tf.expand_dims(clazz, 0)
        return example[DatasetFeature.VIDEO_FACE_RAW.name], clazz

    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        example = tf.io.parse_single_example(serialized_example, self._feature_description)

        video_fragment = tf.io.parse_tensor(example[DatasetFeature.VIDEO_FACE_RAW.name], tf.uint8)
        clazz = tf.io.parse_tensor(example[DatasetFeature.CLASS.name], tf.double)

        clazz = tf.cast(clazz, dtype=tf.float32)
        video_fragment = tf.cast(video_fragment, dtype=tf.float32)

        clazz = tf.ensure_shape(clazz, 9)

        return {
            DatasetFeature.VIDEO_FACE_RAW.name: video_fragment,
            DatasetFeature.CLASS.name: clazz
        }

    @tf.function
    def _C3D_extract_frames_features(self, example: tf.train.Example):
        video_frames = example[DatasetFeature.VIDEO_FACE_RAW.name]

        video_frames = tf.map_fn(self._decode_image, video_frames)
        video_frames = tf.ensure_shape(video_frames, shape=(None, *self._output_shape[1:]))
        video_frames = tf.signal.frame(video_frames, self._output_shape[0], self._frames_step, axis=0)
        frames_features = tf.map_fn(self._extract_r3plus1d_features, video_frames)
        frames_features = tf.expand_dims(frames_features, 0)

        return {DatasetFeature.VIDEO_FACE_RAW.name: frames_features,
                DatasetFeature.CLASS.name: example[DatasetFeature.CLASS.name]}

    @tf.function
    def _decode_image(self, image: tf.Tensor):
        image = tf.ensure_shape(image, shape=(*self._input_face_size, 3))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, tf.constant(self._output_shape[1:3], dtype=tf.int32))
        return image

    @tf.function
    def _extract_r3plus1d_features(self, frame: tf.Tensor):
        # frame shape = (32, 112, 112, 3)
        # net input shape = (1, 3, 32, 112, 112)
        # output shape (1,400)
        resized_frame = tf.experimental.numpy.moveaxis(frame, -1, 0)
        resized_frame = tf.expand_dims(resized_frame, axis=0)
        features = self._pretrained_feature_extractor(resized_frame)['output_0']
        features = tf.squeeze(features)
        return features
