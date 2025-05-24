import tensorflow as tf
import argparse
import os
import json

def combine_model_with_existing(existing_model, new_model):
    """
    Combines two models by averaging their weights.
    
    Args:
        existing_model (tf.keras.Model): The existing model with current weights.
        new_model (tf.keras.Model): The new model with contributed weights.
    
    Returns:
        tf.keras.Model: The combined model with averaged weights.
    """
    combined_weights = []
    existing_weights = existing_model.get_weights()
    new_weights = new_model.get_weights()

    if len(existing_weights) != len(new_weights):
        raise ValueError("The existing model and the new model have different number of layers/weights.")

    for ew, nw in zip(existing_weights, new_weights):
        combined_weights.append((ew + nw) / 2.0)

    combined_model = tf.keras.models.clone_model(existing_model)
    combined_model.set_weights(combined_weights)
    return combined_model

def main(username, projectname, hash_value):
    try:
        # Define paths
        project_dir = os.path.join("users", username, projectname)
        contrib_dir = os.path.join(project_dir, "contrib")
        model_weights_path = os.path.join(project_dir, "model.weights.h5")
        model_config_path = os.path.join(project_dir, "model_config.json")
        contribution_filename = f"{hash_value}.weights.h5"
        new_model_weights_path = os.path.join(contrib_dir, contribution_filename)

        print(f"Project directory: {project_dir}")
        print(f"Contrib directory: {contrib_dir}")
        print(f"Model config path: {model_config_path}")
        print(f"Existing model weights path: {model_weights_path}")
        print(f"Contribution weights path: {new_model_weights_path}")

        # Add debug logging to check if directories and files exist
        print(f"Does project directory exist? {os.path.exists(project_dir)}")
        print(f"Does contrib directory exist? {os.path.exists(contrib_dir)}")
        if os.path.exists(contrib_dir):
            files_in_contrib = os.listdir(contrib_dir)
            print(f"Files in contrib directory: {files_in_contrib}")
        print(f"Does contribution file exist? {os.path.exists(new_model_weights_path)}")

        # Validate paths
        if not os.path.exists(model_config_path):
            print(f"Error: Model configuration file not found at {model_config_path}.")
            return

        if not os.path.exists(new_model_weights_path):
            print(f"Error: Contribution weights file '{contribution_filename}' not found in contrib directory.")
            return

        # Load model configuration
        with open(model_config_path, 'r') as json_file:
            model_json = json_file.read()
        existing_model = tf.keras.models.model_from_json(model_json)
        print("Loaded existing model configuration from JSON.")

        # Load existing model weights
        if os.path.exists(model_weights_path):
            existing_model.load_weights(model_weights_path)
            print(f"Loaded existing model weights from {model_weights_path}.")
        else:
            print(f"No existing weights found at {model_weights_path}. Initializing with random weights.")
            existing_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Load new contribution model weights
        new_model = tf.keras.models.model_from_json(model_json)
        new_model.load_weights(new_model_weights_path)
        print(f"Loaded contribution weights from {new_model_weights_path}.")

        # Combine models
        combined_model = combine_model_with_existing(existing_model, new_model)
        print("Combined the existing model with the new contribution.")

        # Save the combined weights
        combined_model.save_weights(model_weights_path)
        print(f"Saved the combined model weights to {model_weights_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine model contributions in federated learning.")
    parser.add_argument("username", type=str, help="Username directory")
    parser.add_argument("projectname", type=str, help="Project directory")
    parser.add_argument("hash", type=str, help="SHA1 hash of the contribution weights (without .weights.h5)")

    args = parser.parse_args()
    main(args.username, args.projectname, args.hash)