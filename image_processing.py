import tensorflow as tf
import numpy
def extract_labels_from_filename(filename):
    # Remove file extension
    filename = tf.strings.regex_replace(filename, r'\.jpg$', '')
    # Split the filename into parts
    parts = tf.strings.split(filename, '_')

    # Extract the labels
    age = tf.strings.to_number(parts[0], out_type=tf.int32)
    gender = tf.strings.to_number(parts[1], out_type=tf.int32)
    race = tf.strings.to_number(parts[2], out_type=tf.int32)

    return age, gender, race

def load_image_and_labels(filename, image_dir):
    image_path = tf.strings.join([image_dir, filename], separator='/')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [200, 200])
    image = image / 255.0

    age, gender, race = extract_labels_from_filename(filename)

    return image, (age, gender, race)

def data_generator(image_dir, batch_size):
    file_list = tf.data.Dataset.list_files(image_dir + '/*.jpg')
    dataset = file_list.shuffle(buffer_size=1000)

    dataset = dataset.map(lambda filename: load_image_and_labels(filename, image_dir))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset