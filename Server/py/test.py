import argparse
import os
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import sys

def main(username, projectname):
    try:
        print(f"=== Starting test script for {username}/{projectname} ===")
        
        # Define paths - updated for new structure
        project_dir = os.path.join("users", username, projectname)
        model_dir = os.path.join(project_dir, "model")
        model_json_path = os.path.join(model_dir, "model.json")
        
        # Test data is stored in the data directory structure
        # From /py directory, go up one level to Server, then to data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_dir = os.path.dirname(script_dir)
        
        # Based on your specific structure: .../test/test_set/train5/
        test_data_base = os.path.join(server_dir, "data", "users", username, projectname, "test", "test_set", "train5")
        
        accuracy_file_path = os.path.join(project_dir, "accuracy.txt")

        print(f"Final paths:")
        print(f"  Project directory: {project_dir}")
        print(f"  Model directory: {model_dir}")
        print(f"  Model JSON path: {model_json_path}")
        print(f"  Test data directory: {test_data_base}")
        print(f"  Accuracy file path: {accuracy_file_path}")

        # Validate paths
        if not os.path.exists(model_dir):
            print(f"Error: Model directory not found at {model_dir}.")
            return

        if not os.path.exists(model_json_path):
            print(f"Error: Model JSON file not found at {model_json_path}.")
            return

        if not os.path.exists(test_data_base):
            print(f"Error: Test data directory not found at {test_data_base}.")
            print("Let me check alternative paths...")
            
            # Fallback: check if test data is in test_set directly
            alt_path1 = os.path.join(server_dir, "data", "users", username, projectname, "test", "test_set")
            alt_path2 = os.path.join(server_dir, "data", "users", username, projectname, "test")
            
            print(f"Checking alternative path 1: {alt_path1}")
            if os.path.exists(alt_path1):
                items = os.listdir(alt_path1)
                print(f"Items in alt_path1: {items}")
                for item in items:
                    item_path = os.path.join(alt_path1, item)
                    if os.path.isdir(item_path):
                        sub_items = os.listdir(item_path)
                        label_dirs = [f for f in sub_items if os.path.isdir(os.path.join(item_path, f))]
                        if label_dirs:
                            print(f"Found label directories in {item_path}: {label_dirs}")
                            test_data_base = item_path
                            break
            
            print(f"Checking alternative path 2: {alt_path2}")
            if os.path.exists(alt_path2) and test_data_base == os.path.join(server_dir, "data", "users", username, projectname, "test", "test_set", "train5"):
                items = os.listdir(alt_path2)
                print(f"Items in alt_path2: {items}")
            
            if not os.path.exists(test_data_base):
                print(f"Error: Test data directory not found at any expected location.")
                return

        print("=== Loading model ===")
        # Load model from TensorFlow.js format
        print(f"Loading model from TensorFlow.js format: {model_json_path}")
        sys.stdout.flush()
        
        try:
            model = tfjs.converters.load_keras_model(model_json_path)
            print("Successfully loaded model from TensorFlow.js format.")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return

        # Get model input shape
        input_shape = model.input_shape[1:]  # Exclude batch dimension
        print(f"Model input shape: {input_shape}")

        # Determine color mode and image dimensions based on input shape
        if len(input_shape) == 1:
            # Flattened input (like MNIST flattened to 784)
            if input_shape[0] == 784:
                img_height, img_width = 28, 28
                color_mode = 'grayscale'
            else:
                raise ValueError(f"Unsupported flattened input size: {input_shape[0]}")
        elif len(input_shape) == 2:
            # 2D grayscale input
            img_height, img_width = input_shape
            color_mode = 'grayscale'
        elif len(input_shape) == 3:
            # 3D input with channels
            img_height, img_width, channels = input_shape
            if channels == 1:
                color_mode = 'grayscale'
            elif channels == 3:
                color_mode = 'rgb'
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        else:
            raise ValueError(f"Invalid input shape: {input_shape}")

        print(f"Using color mode: {color_mode}")
        print(f"Image target size: ({img_height}, {img_width})")

        print("=== Scanning test data ===")
        sys.stdout.flush()
        
        # Check what test folders exist
        try:
            test_items = os.listdir(test_data_base)
            print(f"All items in test directory: {test_items}")
            
            test_folders = [f for f in test_items if os.path.isdir(os.path.join(test_data_base, f))]
            print(f"Found test label folders: {test_folders}")
            
            # Count images in each folder
            total_images = 0
            for folder in test_folders:
                folder_path = os.path.join(test_data_base, folder)
                images = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                total_images += len(images)
                print(f"Label folder '{folder}': {len(images)} images")
                
        except Exception as e:
            print(f"Error listing test directory: {e}")
            return

        if not test_folders:
            print("Error: No test label folders found.")
            return

        if total_images == 0:
            print("Error: No image files found in test folders.")
            return

        print(f"Total images found: {total_images}")

        print("=== Creating test generator ===")
        sys.stdout.flush()
        
        # Create ImageDataGenerator for loading test images
        datagen = ImageDataGenerator(rescale=1./255)

        try:
            test_generator = datagen.flow_from_directory(
                test_data_base,
                target_size=(img_height, img_width),
                batch_size=32,
                class_mode='categorical',
                color_mode=color_mode,
                shuffle=False
            )
            print(f"Test generator created: {test_generator.samples} samples, {test_generator.num_classes} classes")
        except Exception as e:
            print(f"Error creating test generator: {e}")
            import traceback
            traceback.print_exc()
            return

        print("=== Compiling and evaluating model ===")
        sys.stdout.flush()
        
        # Compile the model (required for evaluation)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model compiled")

        # Evaluate the model
        try:
            print("Starting evaluation...")
            # Calculate the number of steps needed to process all samples
            steps = test_generator.samples // test_generator.batch_size
            if test_generator.samples % test_generator.batch_size != 0:
                steps += 1  # Add one more step for the remaining samples
            
            print(f"Evaluating with {steps} steps for {test_generator.samples} samples")
            loss, accuracy = model.evaluate(test_generator, steps=steps, verbose=1)
            print(f"Evaluation completed. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return

        # Write accuracy to accuracy.txt
        try:
            with open(accuracy_file_path, 'w') as acc_file:
                acc_file.write(f"{accuracy:.4f}")
            print(f"Saved accuracy {accuracy:.4f} to {accuracy_file_path}")
        except Exception as e:
            print(f"Error saving accuracy: {e}")

        print("=== Test completed successfully ===")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model and save accuracy")
    parser.add_argument("username", type=str, help="Username directory")
    parser.add_argument("projectname", type=str, help="Project directory")

    args = parser.parse_args()
    main(args.username, args.projectname)