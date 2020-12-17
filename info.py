import json


def generate_info(local_file_name, df_train, df_val, df_test):
    info = {"train_length": len(df_train), "validation_length": len(df_val),
            "test_length": len(df_test)}

    with open(local_file_name, 'w') as outfile:
        json.dump(info, outfile)
