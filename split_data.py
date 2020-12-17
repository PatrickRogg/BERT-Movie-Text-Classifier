import pandas as pd


TEST_FRACTION = 0.2
TRAIN_FRACTION = 1 - TEST_FRACTION

data = pd.read_csv('data/train.csv')
data.reset_index(inplace=True)
data['sentiment'].replace({'pos': 1, 'neg': 0}, inplace=True)

train_sample = data.sample(frac=TEST_FRACTION, random_state=0)
train_select = train_sample.sample(frac=TRAIN_FRACTION, random_state=0)
train_csv = train_select.values

validate_select = train_sample.drop(index=train_select.index)
validate_csv = validate_select.values

test = pd.read_csv('data/test.csv')
test.reset_index(inplace=True)
test['sentiment'].replace({'pos': 1, 'neg': 0}, inplace=True)
test_csv = test.sample(frac=TEST_FRACTION, random_state=0).values
