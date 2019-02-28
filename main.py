import tensorflow as tf
import json

from data_utils import Data
from models.char_cnn_kim import CharCNNKim

#MainVARs
verbose=1
epochs=5000
batch_size=128
stop_patience=10
config_path="config.json"
training_data_source="data/remo/remo_training_20190227.csv"
validation_data_source="data/remo/remo_testing_20190227.csv"
#model_path='path\to\logs\best_model.h5' #only used if model already trained

if __name__ == "__main__":
    # Load configurations
    config = json.load(open(config_path))
    
    # Load training data
    training_data = Data(data_source=training_data_source,
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()
    
    # Load validation data
    validation_data = Data(data_source=validation_data_source,
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()

    # Structure Model
    model = CharCNNKim(input_size=config["data"]["input_size"],
                       alphabet_size=config["data"]["alphabet_size"],
                       embedding_size=config["char_cnn_kim"]["embedding_size"],
                       conv_layers=config["char_cnn_kim"]["conv_layers"],
                       fully_connected_layers=config["char_cnn_kim"]["fully_connected_layers"],
                       num_of_classes=config["data"]["num_of_classes"],
                       dropout_p=config["char_cnn_kim"]["dropout_p"],
                       optimizer=config["char_cnn_kim"]["optimizer"],
                       loss=config["char_cnn_kim"]["loss"])

    # Train or Load model
    #model.load_model(model_path = model_path)
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=epochs,
                batch_size=batch_size,
                checkpoint_every=config["training"]["checkpoint_every"],
                verbose=verbose,
                stop_patience=stop_patience)

    # Test Model
    model.test(testing_inputs=validation_inputs,
               testing_labels=validation_labels,
               batch_size=batch_size)
               
