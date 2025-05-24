import os
import argparse
import json
import tensorflow as tf

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
        config_values = dict(line.strip().split("=") for line in f if line.strip() and "=" in line)

    print(f"Config values loaded: {config_values}")

    # Get configuration values
    activation_function = config_values.get("activation_function", "relu")
    dropout_rate = float(config_values.get("dropout_rate", "0.2"))
    combining_method = config_values.get("combining_method", "average")
    input_shape = tuple(map(int, config_values.get("input_shape", "28,28").split(",")))
    num_layers = int(config_values.get("num_layers", "3"))
    units_per_layer = int(config_values.get("units_per_layer", "128"))
    num_classes = int(config_values.get("num_classes", "10"))

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

    # Store the weights from the Sequential model
    weights_path = os.path.join(project_dir, "model.weights.h5")
    model.model.save_weights(weights_path)
    print(f"Saved model weights to: {weights_path}")

    # Save the model configuration in JSON format
    config_path = os.path.join(project_dir, "model_config.json")
    model_config = model.model.to_json()
    with open(config_path, "w") as json_file:
        json_file.write(model_config)
    print(f"Saved model config to: {config_path}")
    
    print("Model initialization completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize project with configuration")
    parser.add_argument("username", type=str, help="Username directory")
    parser.add_argument("projectname", type=str, help="Project directory")

    args = parser.parse_args()
    main(args.username, args.projectname)