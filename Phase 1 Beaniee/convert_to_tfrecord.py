import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import tf_record_util
from object_detection.utils import dataset_util
import xml.etree.ElementTree as ET

# Define paths
xml_folder_path = r"C:\Users\Dell\Documents\Hackathon\Phase 1 Beaniee\archive\annotations"
image_folder_path = r"C:\Users\Dell\Documents\Hackathon\Phase 1 Beaniee\archive\images"
output_tfrecord_path = r"C:\Users\Dell\Documents\Hackathon\Phase 1 Beaniee\archive\tfrecord"

# Convert the XML file to tf record
def create_example(image_path, xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    

    # Initialize the image
    image_data = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_data)
    height, width, _ = image.shape

    # Create feature dictionary
    features = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(os.path.basename(image_path).encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(os.path.basename(image_path).encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(image_data),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf-8')),
    }

    # Add object annotations (bounding boxes, labels, etc.)
    xmin, ymin, xmax, ymax, labels = [], [], [], [], []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(label)
        bndbox = obj.find('bndbox')
        xmin.append(int(bndbox.find('xmin').text) / width)
        ymin.append(int(bndbox.find('ymin').text) / height)
        xmax.append(int(bndbox.find('xmax').text) / width)
        ymax.append(int(bndbox.find('ymax').text) / height)

    features.update({
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(labels)
    })

    # Create tf.train.Example
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

# Create TFRecord from images and XML annotations
def create_tfrecord(xml_folder, image_folder, output_tfrecord):
    writer = tf.io.TFRecordWriter(output_tfrecord)

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            # Get corresponding image file name
            image_file = os.path.splitext(xml_file)[0] + '.jpg'
            image_path = os.path.join(image_folder, image_file)
            xml_path = os.path.join(xml_folder, xml_file)

            # Create Example and write to the TFRecord
            example = create_example(image_path, xml_path)
            writer.write(example.SerializeToString())

    writer.close()
    print(f"TFRecord created at: {output_tfrecord}")

# Run the conversion
output_tfrecord_file = os.path.join(output_tfrecord_path, 'output.tfrecord')
create_tfrecord(xml_folder_path, image_folder_path, output_tfrecord_file)
