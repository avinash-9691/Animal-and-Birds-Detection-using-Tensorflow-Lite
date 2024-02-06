import numpy as np
import os
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata
import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)
train_data = object_detector.DataLoader.from_pascal_voc('/content/drive/MyDrive/MCA_final_project/Dataset/TRAIN',
                                                        '/content/drive/MyDrive/MCA_final_project/Dataset/TRAIN',
                                                        ['Other', 'Wild_animal', 'Bird', 'Domestic_animal'])
val_data = object_detector.DataLoader.from_pascal_voc('/content/drive/MyDrive/MCA_final_project/Dataset/TEST',
                                                      '/content/drive/MyDrive/MCA_final_project/Dataset/TEST',
                                                      ['Other', 'Wild_animal', 'Bird', 'Domestic_animal'])
spec = object_detector.EfficientDetSpec(
  model_name='efficientdet-lite0',
  uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
  model_dir='/content/checkpoints',
  hparams={'max_instances_per_image': 8000})
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=10, validation_data=val_data)
eval_result = model.evaluate(val_data)
print("COCO metrics:")
for label, metric_value in eval_result.items():
    print(f"{label}: {metric_value}")
model.export(export_dir='/content/drive/MyDrive/MCA_final_project/Dataset/', tflite_filename='android.tflite')
tflite_eval_result = model.evaluate_tflite('/content/drive/MyDrive/MCA_final_project/Dataset/android.tflite', val_data)

