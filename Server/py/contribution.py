import tensorflow as tf
import tensorflowjs as tfjs
import argparse
import os
import json
import shutil

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

def validate_tfjs_model(model_dir):
    """
    Validates that a TensorFlow.js model directory contains all required files.
    """
    model_json_path = os.path.join(model_dir, 'model.json')
    
    if not os.path.exists(model_json_path):
        return False, "model.json not found"
    
    # Read model.json to find weight files
    try:
        with open(model_json_path, 'r') as f:
            model_data = json.load(f)
        
        weights_manifest = model_data.get('weightsManifest', [])
        if not weights_manifest:
            return False, "No weights manifest found in model.json"
        
        # Check if all weight files exist
        missing_files = []
        for manifest_entry in weights_manifest:
            for weight_path in manifest_entry.get('paths', []):
                weight_file = os.path.join(model_dir, weight_path)
                if not os.path.exists(weight_file):
                    missing_files.append(weight_path)
        
        if missing_files:
            return False, f"Missing weight files: {missing_files}"
        
        return True, "Model is complete"
        
    except Exception as e:
        return False, f"Error reading model.json: {e}"

def main(username, projectname, hash_value):
    try:
        # Define paths
        project_dir = os.path.join("users", username, projectname)
        contrib_dir = os.path.join(project_dir, "contrib")
        model_dir = os.path.join(project_dir, "model")
        contribution_filename = f"{hash_value}.tfjs"
        new_model_dir = os.path.join(contrib_dir, contribution_filename)

        print(f"Project directory: {project_dir}")
        print(f"Contrib directory: {contrib_dir}")
        print(f"Model directory: {model_dir}")
        print(f"Contribution model directory: {new_model_dir}")

        # Add debug logging to check if directories and files exist
        print(f"Does project directory exist? {os.path.exists(project_dir)}")
        print(f"Does contrib directory exist? {os.path.exists(contrib_dir)}")
        if os.path.exists(contrib_dir):
            files_in_contrib = os.listdir(contrib_dir)
            print(f"Files in contrib directory: {files_in_contrib}")
        print(f"Does model directory exist? {os.path.exists(model_dir)}")
        print(f"Does contribution model exist? {os.path.exists(new_model_dir)}")

        # Debug: List files in model directory if it exists
        if os.path.exists(model_dir):
            model_files = os.listdir(model_dir)
            print(f"Files in model directory: {model_files}")
            
            # Validate model completeness
            is_valid, message = validate_tfjs_model(model_dir)
            print(f"Model validation: {message}")
        
        # Debug: List files in contribution directory if it exists
        if os.path.exists(new_model_dir):
            contrib_files = os.listdir(new_model_dir)
            print(f"Files in contribution directory: {contrib_files}")
            
            # Validate contribution completeness
            is_valid, message = validate_tfjs_model(new_model_dir)
            print(f"Contribution validation: {message}")
            
            if not is_valid:
                print(f"Error: Contribution model is incomplete - {message}")
                return

        # Validate contribution exists
        if not os.path.exists(new_model_dir):
            print(f"Error: Contribution model directory '{contribution_filename}' not found in contrib directory.")
            return

        # Check if main model directory exists, if not create it from first contribution
        if not os.path.exists(model_dir):
            print(f"Main model directory not found. Creating initial model from first contribution.")
            os.makedirs(model_dir, exist_ok=True)
            
            # Copy the contribution as the initial model
            for item in os.listdir(new_model_dir):
                src_path = os.path.join(new_model_dir, item)
                dst_path = os.path.join(model_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
            
            print(f"Initialized main model from contribution {hash_value}")
            return

        # Validate main model before loading
        is_valid, message = validate_tfjs_model(model_dir)
        if not is_valid:
            print(f"Main model is incomplete: {message}")
            print("Reinitializing model from current contribution.")
            
            # Replace corrupted model with the contribution
            shutil.rmtree(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            
            for item in os.listdir(new_model_dir):
                src_path = os.path.join(new_model_dir, item)
                dst_path = os.path.join(model_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
            
            print(f"Reinitialized main model from contribution {hash_value}")
            return

        # Load existing model from TensorFlow.js format
        try:
            print(f"Attempting to load existing model from: {model_dir}")
            existing_model = tfjs.converters.load_keras_model(model_dir)
            print("Loaded existing model from TensorFlow.js format.")
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Reinitializing model from current contribution.")
            
            # If main model is corrupted, replace it with the contribution
            shutil.rmtree(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            
            for item in os.listdir(new_model_dir):
                src_path = os.path.join(new_model_dir, item)
                dst_path = os.path.join(model_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
            
            print(f"Reinitialized main model from contribution {hash_value}")
            return

        # Load new contribution model
        try:
            print(f"Attempting to load contribution model from: {new_model_dir}")
            new_model = tfjs.converters.load_keras_model(new_model_dir)
            print(f"Loaded contribution model from {new_model_dir}.")
        except Exception as e:
            print(f"Error loading contribution model: {e}")
            return

        # Combine models
        try:
            combined_model = combine_model_with_existing(existing_model, new_model)
            print("Combined the existing model with the new contribution.")

            # Save the combined model in TensorFlow.js format
            # Remove existing model directory and save new combined model
            shutil.rmtree(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            tfjs.converters.save_keras_model(combined_model, model_dir)
            print(f"Saved the combined model to {model_dir}.")
        except Exception as e:
            print(f"Error combining models: {e}")
            return

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine model contributions in federated learning.")
    parser.add_argument("username", type=str, help="Username directory")
    parser.add_argument("projectname", type=str, help="Project directory")
    parser.add_argument("hash", type=str, help="SHA1 hash of the contribution weights (without .weights.h5)")

    args = parser.parse_args()
    main(args.username, args.projectname, args.hash)