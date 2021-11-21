Object Detection with TensorFlow
================================

Train the model as per [Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#). Stop when total loss is somewhere less than 1.

```
python scripts\training\model_main_tf2.py --model_dir=models\ssd_mobilenet_v2_320x320_ball_quant --pipeline_config_path=models\ssd_mobilenet_v2_320x320_ball_quant\pipeline.config
```

Export to Saved Model for local testing

```
python scripts\training\exporter_main_v2.py -input_type image_tensor --pipeline_config_path models\ssd_mobilenet_v2_320x320_ball\pipeline.config --trained_checkpoint_dir .\models\ssd_mobilenet_v2_320x320_ball --output_directory .\exported-models\ssd_mobilenet_v2_320x320_ball
```

Export to TF Lite for Raspberry Pi as per [Running TF2 Detection API Models on mobile](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)

```
python scripts\training\export_tflite_graph_tf2.py --pipeline_config_path models\ssd_mobilenet_v2_320x320_ball\pipeline.config --trained_checkpoint_dir models\ssd_mobilenet_v2_320x320_ball --output_directory exported-models\ssd_mobilenet_v2_320x320_ball\tflite_graph
```

```
python scripts\training\convert_tflite_graph_tf2.py
```
