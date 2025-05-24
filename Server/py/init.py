import os
import argparse
import json
import tensorflow as tf
import tensorflowjs as tfjs

class YourNetworkClass(tf.keras.Model):
    def __init__(self, input_shape, activation_function, dropout_rate, num_layers, units_per_layer, num_classes):
        super(YourNetworkClass, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        self.model.add(tf.keras.layers.Flatten())
        for _ in range(num_layers):
            self.model.add(tf.keras.layers.Dense(units_per_layer, activation=activation_function))
            self.model.add(tf.keras.layers.Dropout(dropout_rate))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, inputs):
        return self.model(inputs)

    def initialize_weights(self):
        # Initialize weights by running a forward pass with dummy data
        input_shape_with_batch = [1] + list(self.model.input_shape[1:])
        dummy_input = tf.random.normal(input_shape_with_batch)
        self.call(dummy_input)

def main(username, projectname):
    project_dir = os.path.join("users", username, projectname)
    contrib_dir = os.path.join(project_dir, "contrib")

    # Create project and contrib directories if they don't exist
    os.makedirs(contrib_dir, exist_ok=True)
    
    print(f"Project directory: {project_dir}")
    print(f"Contrib directory: {contrib_dir}")

    # Read values from config.txt in the data folder structure
    config_file = os.path.join("..", "data", "users", username, projectname, "py", "config.txt")
    print(f"Looking for config file at: {config_file}")
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return
    
    with open(config_file, "r") as f:
        config_values = dict(line.strip().split("=", 1) for line in f if line.strip() and "=" in line)

    print(f"Config values loaded: {config_values}")

    # Get configuration values with better handling of empty/invalid values
    def get_config_value(key, default, value_type=str):
        value = config_values.get(key, default)
        if not value or value.strip() == "":
            return default if isinstance(default, value_type) else value_type(default)
        try:
            return value_type(value.strip())
        except (ValueError, TypeError):
            print(f"Warning: Invalid value '{value}' for {key}, using default {default}")
            return default if isinstance(default, value_type) else value_type(default)

    activation_function = get_config_value("activation_function", "relu", str)
    dropout_rate = get_config_value("dropout_rate", 0.2, float)
    combining_method = get_config_value("combining_method", "average", str)
    
    # Handle input_shape specially
    input_shape_str = get_config_value("input_shape", "28,28", str)
    try:
        input_shape = tuple(map(int, input_shape_str.split(",")))
    except (ValueError, AttributeError):
        print(f"Warning: Invalid input_shape '{input_shape_str}', using default (28,28)")
        input_shape = (28, 28)
    
    num_layers = get_config_value("num_layers", 3, int)
    units_per_layer = get_config_value("units_per_layer", 128, int)
    num_classes = get_config_value("num_classes", 10, int)

    # Adjust input_shape to include channels dimension if missing
    if len(input_shape) == 2:
        input_shape += (1,)  # Add channel dimension for grayscale images

    print(f"Creating model with: input_shape={input_shape}, layers={num_layers}, units={units_per_layer}")

    # Initialize the required network with random weights
    model = YourNetworkClass(
        input_shape, activation_function, dropout_rate,
        num_layers, units_per_layer, num_classes
    )
    model.initialize_weights()

    # Save the model in TensorFlow.js format
    model_dir = os.path.join(project_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    tfjs.converters.save_keras_model(model.model, model_dir)
    print(f"Saved model in TensorFlow.js format to: {model_dir}")
    
    print("Model initialization completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize project with configuration")
    parser.add_argument("username", type=str, help="Username directory")
    parser.add_argument("projectname", type=str, help="Project directory")

    args = parser.parse_args()
    main(args.username, args.projectname)