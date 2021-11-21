import cv2
import numpy as np
import sys
import tensorflow as tf
import time

# Function to draw a rectangle with width > 1
def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline = color, fill = color)

def apply_non_max_suppression(detections, iou_thresh=0.5, score_thresh=0.6):
    """
    Function to apply non-maximum suppression on different classes
    Parameters
    ----------
    detections : dictionary
        dictionary containing:
            'detection_boxes' : Bounding boxes coordinates. Shape (N, 4)
            'detection_classes' : Class indices detected. Shape (N)
            'detection_scores' : Shape (N)
            'num_detections' : Total number of detections i.e. N. Shape (1)
    iou_thresh : int, optional
        Intersection Over Union threshold value. The default is 0.5.
    score_thresh : int, optional
        Score threshold value below which to ignore. The default is 0.6.
    Returns
    -------
    detections : dictionary
        dictionary containing only scores and IOU greater than threshold.
            'detection_boxes' : Bounding boxes coordinates. Shape (N2, 4)
            'detection_classes' : Class indices detected. Shape (N2)
            'detection_scores' : Shape (N2)
            where N2 is the number of valid predictions after those conditions.
    """

    q = 90 # no of classes in model
    num = int(detections['num_detections'])
    boxes = np.zeros([1, num, q, 4])
    scores = np.zeros([1, num, q])
    # val = [0]*q
    for i in range(num):
        # indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
        boxes[0, i, detections['detection_classes'][i]-1, :] = detections['detection_boxes'][i]
        scores[0, i, detections['detection_classes'][i]-1] = detections['detection_scores'][i]

    nmsd = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size_per_class=num,
        max_total_size=num,
        iou_threshold=iou_thresh,
        score_threshold=score_thresh,
        pad_per_class=False,
        clip_boxes=False)
    valid = nmsd.valid_detections[0].numpy()
    return {
        'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
        'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
        'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
    }


print('Loading model...', end='')
start_time = time.time()

# Function to read labels from text files.
label_file = './exported-models/ssd_mobilenet_v2_320x320_ball_tflite/tflite_label_map.txt'
with open(label_file, 'r') as f:
    lines = f.readlines()
labels = {}
for line in lines:
    pair = line.strip().split(maxsplit=1)
    labels[int(pair[0])] = pair[1].strip()

# Load TFLite model and allocate tensors.
model_file = './exported-models/ssd_mobilenet_v2_320x320_ball_tflite/model_with_metadata.tflite'
interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Done! Took {elapsed_time} seconds')

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = False
if input_details[0]['dtype'] == np.float32:
    floating_model = True

# print("Input details: ", input_details)
# print("Output details: ", output_details)

# These seem to be in the reverse order specified in 
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md
# *shrug*
prefix = 'StatefulPartitionedCall:'
detection_boxes_index = next(x for x in output_details if x['name'] == prefix + '3')['index']
detection_classes_index = next(x for x in output_details if x['name'] == prefix + '2')['index']
detection_scores_index = next(x for x in output_details if x['name'] == prefix + '1')['index']
num_boxes_index = next(x for x in output_details if x['name'] == prefix + '0')['index']

cap = cv2.VideoCapture(sys.argv[1])
codec = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter("tracking-output.avi", codec, 25, (320,240))

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 255, 0)

frameCount = 0
start_time = time.time()
while(True):
    ret, image = cap.read()
    if not ret:
        break
   
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = False
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    initial_h, initial_w, channels = image.shape
    frame = cv2.resize(image, (width, height))

    # add N dim
    input_data = np.expand_dims(frame, axis=0)
  
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    detection_boxes = interpreter.get_tensor(detection_boxes_index)[0]
    detection_classes = interpreter.get_tensor(detection_classes_index)[0].astype(np.int64)
    detection_scores = interpreter.get_tensor(detection_scores_index)[0]
    num_boxes = int(interpreter.get_tensor(num_boxes_index)[0])

    # print("num_boxes:", num_boxes)
    # print("detected boxes:", detection_boxes)
    # print("detected classes:", detection_classes)
    # print("detected scores:", detection_scores)

    for i in range(num_boxes):
        top, left, bottom, right = detection_boxes[i]
        classId = int(detection_classes[i])
        score = detection_scores[i]
        if score > 0.5:
            xmin = int(left * initial_w)
            ymin = int(top * initial_h)
            xmax = int(right * initial_w)
            ymax = int(bottom * initial_h)
            box = [xmin, ymin, xmax, ymax]
            print(labels[classId], 'score:', score, ', box:', box)
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 2)
            result_text = labels[classId] + ' ' + str(round(score*100, 1)) + '%'
            text_location = (_MARGIN + xmin, _MARGIN + _ROW_SIZE + ymax)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    out.write(image)
    frameCount +=1 
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Done! {frameCount} frames took {elapsed_time} seconds, or {frameCount/elapsed_time} fps')

out.release()
cap.release()
cv2.destroyAllWindows()
