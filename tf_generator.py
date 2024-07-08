import os
import tensorflow as tf
from object_detection.utils import dataset_util
from lxml import etree
import io
from PIL import Image

def create_tf_example(image_path, annotation_path):
    with tf.io.gfile.GFile(annotation_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    img_path = image_path
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = data['filename'].encode('utf8')
    image_format = b'jpg'  # or b'png'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in data['object']:
        xmin = float(obj['bndbox']['xmin']) / width
        xmax = float(obj['bndbox']['xmax']) / width
        ymin = float(obj['bndbox']['ymin']) / height
        ymax = float(obj['bndbox']['ymax']) / height
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(1)  # Assuming 'person' class is indexed as 1

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    # Especifica las rutas aquí
    #train_images_dir = 'InriData/images/train'
    #train_annotations_dir = 'InriData/annotations/train'
    #train_output_path = 'InriData/train.record'
    train_images_dir = 'InriData/Train/JPEGImages'
    train_annotations_dir = 'InriData/Train/Annotations'
    train_output_path = 'InriData/Train/Records/train.record'

    # test_images_dir = 'InriData/images/test'
    # test_annotations_dir = 'InriData/annotations/test'
    # test_output_path = 'InriData/test.record'
    test_images_dir = 'InriData/Test/JPEGImages'
    test_annotations_dir = 'InriData/Test/Annotations'
    test_output_path = 'InriData/Test/Records/test.record'

    # Convierte y guarda los archivos TFRecord para el conjunto de entrenamiento
    with tf.io.TFRecordWriter(train_output_path) as writer:
        for filename in os.listdir(train_images_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(train_images_dir, filename)
                annotation_path = os.path.join(train_annotations_dir, os.path.splitext(filename)[0] + '.xml')
                tf_example = create_tf_example(image_path, annotation_path)
                writer.write(tf_example.SerializeToString())

    # Convierte y guarda los archivos TFRecord para el conjunto de prueba
    with tf.io.TFRecordWriter(test_output_path) as writer:
        for filename in os.listdir(test_images_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(test_images_dir, filename)
                annotation_path = os.path.join(test_annotations_dir, os.path.splitext(filename)[0] + '.xml')
                tf_example = create_tf_example(image_path, annotation_path)
                writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
    main()