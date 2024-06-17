import numpy as np
import tensorflow as tf
import cv2
import os
import tarfile
import urllib.request

def download_and_extract_model(model_url, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_filename = model_url.split('/')[-1]
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):
        print("Downloading model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")
    else:
        print("Model already exists!")

    print("Extracting model...")
    with tarfile.open(model_path, 'r') as tar:
        tar.extractall(model_dir)
    print("Model extracted successfully!")

model_url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz'
model_dir = 'ssd_mobilenet_v2_coco_2018_03_29'
#download_and_extract_model(model_url, model_dir)

model = tf.saved_model.load(os.path.join(model_dir + '\\' + model_dir, 'saved_model'))
print(model.signatures)

cap = cv2.VideoCapture(0)  # Usar la cámara predeterminada

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))
    input_tensor = tf.convert_to_tensor(image_resized)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    detections = model.signatures['serving_default'](input_tensor)
    detection_classes = detections['detection_classes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_boxes = detections['detection_boxes'][0].numpy()
    num_detections = int(detections['num_detections'][0].numpy())

    people_counter = 0
    h, w, _ = frame.shape

    for i in range(num_detections):
        class_id = int(detection_classes[i])
        score = detection_scores[i]
        if class_id == 1 and score > 0.93:  # Ajusta el umbral según sea necesario
            ymin, xmin, ymax, xmax = detection_boxes[i]
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            people_counter += 1

    print(f"Personas detectadas: {people_counter}")
    cv2.imshow('Detected People', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
image = cv2.imread('image2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_resized = cv2.resize(image_rgb, (300, 300))
input_tensor = tf.convert_to_tensor(image_resized)
input_tensor = tf.expand_dims(input_tensor, axis=0)

detections = model.signatures['serving_default'](input_tensor)
detection_classes = detections['detection_classes'][0]
people_counter = 0

i = 0
for detection in detections['detection_boxes'][0]:
    ymin, xmin, ymax, xmax = detection
    class_id = int(detection_classes[i])
    if class_id == 1 and detections['detection_scores'][0][0] > 0.93:  # Adjust threshold as needed
        h, w, _ = image.shape
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        people_counter += 1
    i += 1


print(int(detections['num_detections'][0]))

cv2.imshow('Detected People', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''