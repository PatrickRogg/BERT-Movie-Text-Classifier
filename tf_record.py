import time
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordWriter


def create_tf_example(features, label):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[0]])),
        'sentence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[1].encode('utf-8')])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))
    return tf_example.SerializeToString()


def convert_csv_to_tfrecord(csv, file_name):
    start_time = time.time()
    writer = TFRecordWriter(file_name)
    for idx, row in enumerate(csv):
        try:
            if row is None:
                raise Exception('Row Missing')
            if row[0] is None or row[1] is None or row[2] is None:
                raise Exception('Value Missing')
            if row[1].strip() == '':
                raise Exception('Utterance is empty')

            features, label = row[:-1], row[-1]
            example = create_tf_example(features, label)
            writer.write(example)

        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
    writer.close()
    print(f"{file_name}: --- {(time.time() - start_time)} seconds ---")