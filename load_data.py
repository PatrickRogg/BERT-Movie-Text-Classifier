import tensorflow as tf

feature_spec = {
    'idx': tf.io.FixedLenFeature([], tf.int64),
    'sentence': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


def load_tf_record(tf_record_path):
    tf_record = tf.data.TFRecordDataset(tf_record_path)
    tf_parsed_record = tf_record.map(parse_example)
    tf_cleaned_record = tf_parsed_record.map(lambda features: clean_string(features))
    return tf_cleaned_record


def clean_string(features):
    revised_sentence = tf.strings.regex_replace(features['sentence'], "\.\.\.", "", replace_global=True)
    revised_sentence = tf.strings.regex_replace(revised_sentence, "\\'", "'", replace_global=True)
    revised_sentence = tf.strings.regex_replace(revised_sentence, "\\n", "", replace_global=True)
    features['sentence'] = revised_sentence
    return features


def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_spec)
