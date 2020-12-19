from info import generate_info
from load_data import load_tf_record
from split_data import train_csv, validate_csv, test_csv
from tf_record import convert_csv_to_tfrecord
from train import compile, evaluate, fit

BATCH_SIZE = 8
EVAL_BATCH_SIZE = BATCH_SIZE * 2

convert_csv_to_tfrecord(train_csv, "data/movie_train.tfrecord")
convert_csv_to_tfrecord(validate_csv, "data/movie_validate.tfrecord")
convert_csv_to_tfrecord(test_csv, "data/movie_test.tfrecord")

generate_info('./data/info.json', train_csv, validate_csv, test_csv)

train_data = load_tf_record("data/movie_train.tfrecord")
validate_data = load_tf_record("data/movie_validate.tfrecord")
test_data = load_tf_record("data/movie_test.tfrecord")

compiled_validate_data = compile(validate_data).batch(BATCH_SIZE)
compiled_training_data = compile(train_data).shuffle(len(train_csv)).batch(BATCH_SIZE).repeat(-1)
compiled_test_data = compile(test_data).batch(BATCH_SIZE)

fit(compiled_training_data, compiled_validate_data)

evaluate(compiled_test_data)

