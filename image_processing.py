import tensorflow as tf

def extract_labels_from_filename(filename):
    # Remove file extension
    filename = tf.strings.regex_replace(filename, r'\.jpg\.chip\.jpg$', '')
    # Split the filename into parts
    parts = tf.strings.split(filename, '_')

    # Extract the labels
    age = tf.strings.to_number(parts[0], out_type=tf.int32)
    gender = tf.strings.to_number(parts[1], out_type=tf.int32)
    race = tf.strings.to_number(parts[2], out_type=tf.int32)

    return age, gender, race

def load_image_and_labels(filename):
    # Replace backward slashes with forward slashes
    filename = tf.strings.regex_replace(filename, '\\\\', '/')

    # Load and decode the image
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize the image to 200x200 pixels
    image = tf.image.resize(image, [200, 200])

    # Normalize the image to range [0, 1]
    image = image / 255.0

    # Extract labels from the filename
    age, gender, race = extract_labels_from_filename(tf.strings.split(filename, '/')[-1])

    return image, (age, gender, race)

def data_generator(image_dir, batch_size):
    # Create a dataset of file paths
    file_list = tf.data.Dataset.list_files(image_dir + '/*.jpg')

    # Shuffle the dataset
    dataset = file_list.shuffle(buffer_size=1000)

    # Map filenames to images and labels
    dataset = dataset.map(load_image_and_labels)

    # Batch the data and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
