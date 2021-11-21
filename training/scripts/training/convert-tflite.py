import cv2
import glob
import numpy as np
import pathlib
import tensorflow as tf
from object_detection.utils import label_map_util
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

_TFLITE_MODEL_PATH = "exported-models/ssd_mobilenet_v2_320x320_ball_tflite/model_quant.tflite"

def representative_dataset():
  a = []
  image_files = glob.glob('./images/*.jpg')
  # for image_file in np.random.choice(image_files, 100):
  for image_file in image_files:
      img = cv2.imread(image_file)
      img = cv2.resize(img, (300, 300))
      img = img.astype(np.float32)
      img = img / 255.0
      a.append(img)
  a = np.array(a)
  print(a.shape) # a is np array of 160 3D images
  # for i in tf.data.Dataset.from_tensor_slices(a).batch(1).take(100):
  for i in tf.data.Dataset.from_tensor_slices(a).batch(1):
    # print(i.shape)
    yield [i]

def representative_dataset2():
  train_ds = tf.keras.utils.image_dataset_from_directory(
      pathlib.Path('./images/all'),
      labels=None,
      label_mode=None,
      image_size=(300, 300),
      batch_size=200)
  # images = tf.cast(train_ds[0], tf.float32) / 255.0
  # train_ds_tensors = tf.data.Dataset.from_tensor_slices((train_ds)).batch(1)
  # print(train_ds_tensors)
  # dataset = tf.data.TFRecordDataset('annotations/train.record')
  # print(dataset)
  for input_value in train_ds:
    yield [input_value]
  # for image_batch in train_ds_tensors:
  #   yield [image_batch]

# def representative_dataset():
#   for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
#     yield [tf.dtypes.cast(data, tf.float32)]


def representative_dataset3():
  """ it yields an image one by one """
# create an image generator with a batch size of 1 
  test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
  test_generator = test_datagen.flow_from_directory(
      './images', 
      target_size=(300, 300), 
      batch_size=1,
      classes=['all'],
      class_mode='categorical')
  for ind in range(len(test_generator.filenames)):
    img_with_label = test_generator.next() # it returns (image and label) tuple
    yield [np.array(img_with_label[0], dtype=np.float32, ndmin=2)] # return only image


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('exported-models/ssd_mobilenet_v2_320x320_ball_tflite/saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset3
# Set the input and output tensors to uint8
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the model.
with open(_TFLITE_MODEL_PATH, 'wb') as f:
  f.write(tflite_model)

# We need to convert the Object Detection API's labelmap into what the Task API needs:
# a txt file with one class name on each line from index 0 to N.
# The first '0' class indicates the background.
# This code assumes COCO detection which has 90 classes, you can write a label
# map file for your model if re-trained.
_ODT_LABEL_MAP_PATH = 'annotations/ball_label_map.pbtxt'
_TFLITE_LABEL_PATH = "exported-models/ssd_mobilenet_v2_320x320_ball_tflite/tflite_label_map.txt"

category_index = label_map_util.create_category_index_from_labelmap(
    _ODT_LABEL_MAP_PATH)
f = open(_TFLITE_LABEL_PATH, 'w')
for class_id in range(1, 2):
  if class_id not in category_index:
    f.write('???\n')
    continue
  name = category_index[class_id]['name']
  f.write(str(class_id-1) + ' ' + name +'\n')
f.close()

_TFLITE_MODEL_WITH_METADATA_PATH = "exported-models/ssd_mobilenet_v2_320x320_ball_tflite/model_quant_with_metadata.tflite"

writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file(_TFLITE_MODEL_PATH), input_norm_mean=[127.5], 
    input_norm_std=[127.5], label_file_paths=[_TFLITE_LABEL_PATH])
writer_utils.save_file(writer.populate(), _TFLITE_MODEL_WITH_METADATA_PATH)

displayer = metadata.MetadataDisplayer.with_model_file(_TFLITE_MODEL_WITH_METADATA_PATH)
print("Metadata populated:")
print(displayer.get_metadata_json())
print("=============================")
print("Associated file(s) populated:")
print(displayer.get_packed_associated_file_list())
