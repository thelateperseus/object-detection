Object Detection with TensorFlow
================================

Partition the image dataset into training and test:

```
python scripts\preprocessing\partition_dataset.py -x -i images -r 0.1
```

Create TensorFlow records for the training and test data:

```
python scripts\preprocessing\generate_tfrecord.py -x images\train -l annotations\label_map.pbtxt -o annotations\train.record -c annotations\train.csv
python scripts\preprocessing\generate_tfrecord.py -x images\test -l annotations\label_map.pbtxt -o annotations\test.record -c annotations\test.csv
```

Train the model as per [Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#). Stop when total loss is somewhere less than 1.

```
python scripts\training\model_main_tf2.py --model_dir=models\ssd_mobilenet_v2_320x320_ball2 --pipeline_config_path=models\ssd_mobilenet_v2_320x320_ball2\pipeline.config
```

Export to Saved Model for local testing

```
python scripts\training\exporter_main_v2.py -input_type image_tensor --pipeline_config_path models\ssd_mobilenet_v2_320x320_ball2\pipeline.config --trained_checkpoint_dir .\models\ssd_mobilenet_v2_320x320_ball2 --output_directory .\exported-models\ssd_mobilenet_v2_320x320_ball2
```

Export to TF Lite for Raspberry Pi as per [Running TF2 Detection API Models on mobile](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)

```
python scripts\training\export_tflite_graph_tf2.py --pipeline_config_path models\ssd_mobilenet_v2_320x320_ball\pipeline.config --trained_checkpoint_dir models\ssd_mobilenet_v2_320x320_ball --output_directory exported-models\ssd_mobilenet_v2_320x320_ball\tflite_graph
```

```
python scripts\training\convert_tflite_graph_tf2.py
```
