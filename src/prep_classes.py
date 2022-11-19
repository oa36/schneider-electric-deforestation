import pandas as pd
import os

from config.models_config import labels_category

def prep_classes(path, type = "train"):
    labels_df = pd.read_csv(path)
    labels_df["example_path"] = labels_df["example_path"].apply(lambda row: os.path.join("./data/", row))
    
    if type == "train":
        train_df = labels_df.sample(frac=0.85, random_state=1234)
        train_df.to_pickle("intermediate_outputs/train_labels.pickle")
        
        val_df = labels_df.drop(train_df.index)

        val_df.to_pickle("intermediate_outputs/val_labels.pickle")
    else:
        labels_df.to_pickle("intermediate_outputs/" + type + "_labels.pickle")

if __name__ == "__main__":
    if not os.path.exists('intermediate_outputs'):
        os.makedirs('intermediate_outputs')
    
    data_path = "./data/"
    path_train = os.path.join(data_path, "train.csv")
    path_test = os.path.join(data_path, "test.csv")
    
    prep_classes(path_train, type = "train")
    prep_classes(path_test, type = "test")
    