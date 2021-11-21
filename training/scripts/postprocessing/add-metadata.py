from object_detection.utils import label_map_util
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

_TFLITE_MODEL_PATH = 'C:/Users/Nathan/Documents/Dev/benchmarking_edge_computing/ssd_mobilenet_v2/tflite_for_rpi/mobilenet_v2.tflite'
_TFLITE_LABEL_PATH = "C:/Users/Nathan/Documents/Dev/benchmarking_edge_computing/ssd_mobilenet_v2/tflite_for_rpi/coco_labels.txt"
_TFLITE_MODEL_WITH_METADATA_PATH = 'C:/Users/Nathan/Documents/Dev/benchmarking_edge_computing/ssd_mobilenet_v2/tflite_for_rpi/mobilenet_v2_metadata.tflite'

writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file(_TFLITE_MODEL_PATH), input_norm_mean=[127.5], 
    input_norm_std=[127.5], label_file_paths=[_TFLITE_LABEL_PATH])
writer_utils.save_file(writer.populate(), _TFLITE_MODEL_WITH_METADATA_PATH)
