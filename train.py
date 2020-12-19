import time
import json
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features

USE_XLA = False
USE_AMP = False

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

num_labels = 2
config = BertConfig.from_pretrained("bert-base-cased", num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', config=config)


def compile(data):
    start_time = time.time()
    trained_data = glue_convert_examples_to_features(examples=data, tokenizer=tokenizer, max_length=512, task='sst-2',
                                                     label_list=['0', '1'])
    print(f"---{time.time() - start_time} seconds---")
    opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)

    if USE_AMP:
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=opt, loss=loss, metrics=[metric])
    model.summary()
    return trained_data


def fit(train_dataset, valid_dataset):
    train_steps, valid_steps, test_steps = get_info()
    history = model.fit(train_dataset, epochs=3, steps_per_epoch=train_steps,
                        validation_data=valid_dataset, validation_steps=valid_steps)
    print(history)


def evaluate(test_data):
    model.evaluate(test_data)


def get_info():
    with open('data/info.json') as json_file:
        data_info = json.load(json_file)

    train_steps = data_info['train_length']
    valid_steps = data_info['validation_length']
    test_steps = data_info['test_length']

    return train_steps, valid_steps, test_steps
