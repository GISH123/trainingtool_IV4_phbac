import tensorflow as tf
from configuration import train_dir, valid_dir, test_dir, train_tfrecord, valid_tfrecord, test_tfrecord
from prepare_data import get_images_and_labels
import random

# convert a value to a type compatible tf.train.Feature
def _bytes_feature(value):
    # Returns a bytes_list from a string / byte.
    if isinstance(value, type(tf.constant(0.))):
        value = value.numpy()   # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    # Returns a float_list from a float / double.
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    # Returns an int64_list from a bool / enum / int / uint.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _image_path_to_bytes(image_path):
    return image_path.encode('utf-8')

# Create a dictionary with features that may be relevant.
def image_example(image_path, image_string, label):
    image_path_bytes = _image_path_to_bytes(image_path)
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
        'image_path': _bytes_feature(image_path_bytes),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def shuffle_dict(original_dict):
    keys = []
    shuffled_dict = {}
    for k in original_dict.keys():
        keys.append(k)
    random.shuffle(keys)
    for item in keys:
        shuffled_dict[item] = original_dict[item]
    return shuffled_dict


def dataset_to_tfrecord(dataset_dir, tfrecord_name):
    image_paths, image_labels = get_images_and_labels(dataset_dir)
    image_paths_and_labels_dict = {}
    for i in range(len(image_paths)):
        image_paths_and_labels_dict[image_paths[i]] = image_labels[i]
    """
    image_paths_and_labels_dict :
    {
        "dataset/train/club_01_ace/xxx1.jpg" : 1, 
        "dataset/train/club_01_ace/xxx2.jpg" : 1, 
        ...
        "dataset/train/diamond_05/xxx153.jpg" : 18, 
        ...
    }
    """
    # shuffle the dict
    image_paths_and_labels_dict = shuffle_dict(image_paths_and_labels_dict)
    # write the images and labels to tfrecord format file
    with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
        for image_path, label in image_paths_and_labels_dict.items():
            print("Writing to tfrecord: {}".format(image_path))
            image_string = open(image_path, 'rb').read()
            tf_example = image_example(image_path, image_string, label)
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    dataset_to_tfrecord(dataset_dir=train_dir, tfrecord_name=train_tfrecord)
    dataset_to_tfrecord(dataset_dir=valid_dir, tfrecord_name=valid_tfrecord)
    dataset_to_tfrecord(dataset_dir=test_dir, tfrecord_name=test_tfrecord)