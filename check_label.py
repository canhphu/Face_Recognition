import os
import tensorflow as tf
import csv

def extract_labels_from_filename(filename):
    # Remove file extension
    filename = tf.strings.regex_replace(filename, r'\.jpg$', '')
    # Split the filename into parts
    parts = tf.strings.split(filename, '_')

    if len(parts) != 4:
        return None, None, None

    # Extract the labels
    try:
        age = tf.strings.to_number(parts[0], out_type=tf.int32)
        gender = tf.strings.to_number(parts[1], out_type=tf.int32)
        race = tf.strings.to_number(parts[2], out_type=tf.int32)
    except ValueError:
        return None, None, None

    return age, gender, race

def is_valid_age(age):
    return 0 <= age <= 116

def is_valid_gender(gender):
    return gender in [0, 1]

def is_valid_race(race):
    return race in [0, 1, 2, 3, 4]

def check_labels(image_dir, output_file):
    # Open CSV file to write invalid filenames
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Error Type'])

        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    tf_filename = tf.strings.as_string(file)
                    age, gender, race = extract_labels_from_filename(tf_filename)

                    if age is None or gender is None or race is None:
                        writer.writerow([file, 'Invalid format'])
                    else:
                        if not is_valid_age(age):
                            writer.writerow([file, 'Invalid age'])
                        if not is_valid_gender(gender):
                            writer.writerow([file, 'Invalid gender'])
                        if not is_valid_race(race):
                            writer.writerow([file, 'Invalid race'])

if __name__ == "__main__":
    image_dir = r"C:\Users\admin\PycharmProjects\Face Recognition\train_data"
    output_file = 'invalid_labels.csv'
    check_labels(image_dir, output_file)
    print(f"Check complete. Invalid labels are listed in {output_file}.")
