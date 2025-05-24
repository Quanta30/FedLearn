import kagglehub
import os
import shutil
import random
from pathlib import Path

def create_federated_dataset():
    # Download latest version of MNIST as JPG
    print("Downloading MNIST dataset...")
    path = kagglehub.dataset_download("scolianni/mnistasjpg")
    print("Path to dataset files:", path)
    
    # Create training_data directory
    training_data_dir = Path("training_data")
    if training_data_dir.exists():
        shutil.rmtree(training_data_dir)
    training_data_dir.mkdir()
    
    # More robust search for training data
    source_train_dir = None
    possible_paths = [
        Path(path) / "trainingSet",
        Path(path) / "train", 
        Path(path) / "training",
        Path(path) / "mnist_train",
        Path(path)  # Check root directory too
    ]
    
    # First try known paths
    for test_path in possible_paths:
        if test_path.exists():
            # Check if this directory contains digit subdirectories
            digit_dirs = [d for d in test_path.iterdir() if d.is_dir() and d.name.isdigit()]
            if digit_dirs:
                source_train_dir = test_path
                break
    
    # If still not found, do a deeper search
    if source_train_dir is None:
        print("Searching for training data...")
        def find_digit_directories(base_path, max_depth=3):
            if max_depth <= 0:
                return None
            
            for item in base_path.iterdir():
                if item.is_dir():
                    # Check if this directory contains digit subdirectories
                    digit_dirs = [d for d in item.iterdir() if d.is_dir() and d.name.isdigit()]
                    if len(digit_dirs) >= 5:  # Should have at least 5 digit directories for MNIST
                        return item
                    
                    # Recursively search subdirectories
                    result = find_digit_directories(item, max_depth - 1)
                    if result:
                        return result
            return None
        
        source_train_dir = find_digit_directories(Path(path))
    
    if source_train_dir is None:
        # List all contents for debugging
        print(f"Could not find training data. Contents of {path}:")
        for item in Path(path).rglob("*"):
            if item.is_dir():
                print(f"  Directory: {item}")
                # Check first few items in each directory
                items = list(item.iterdir())[:5]
                for subitem in items:
                    print(f"    {subitem}")
                if len(items) > 5:
                    print(f"    ... and {len(list(item.iterdir())) - 5} more items")
        raise FileNotFoundError(f"Could not find training data directory with digit subdirectories in {path}")
    
    print(f"Using training data from: {source_train_dir}")
    
    # Get all digit directories (0-9)
    digit_dirs = [d for d in source_train_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    digit_dirs.sort(key=lambda x: int(x.name))
    
    print(f"Found digit directories: {[d.name for d in digit_dirs]}")
    
    if not digit_dirs:
        print("No digit directories found. Listing directory contents:")
        for item in source_train_dir.iterdir():
            print(f"  {item.name} ({'directory' if item.is_dir() else 'file'})")
        raise ValueError("No digit directories found in training data")
    
    # Create 5 training partitions
    num_partitions = 5
    for i in range(1, num_partitions + 1):
        partition_dir = training_data_dir / f"train{i}"
        partition_dir.mkdir()
        print(f"Creating partition {i}...")
        
        # For each digit, copy a subset of images to this partition
        for digit_dir in digit_dirs:
            digit_name = digit_dir.name
            
            # Create digit directory in partition
            partition_digit_dir = partition_dir / digit_name
            partition_digit_dir.mkdir()
            
            # Get all images for this digit
            image_files = list(digit_dir.glob("*.jpg")) + list(digit_dir.glob("*.png"))
            
            # Shuffle images for random distribution
            random.shuffle(image_files)
            
            # Calculate how many images this partition should get
            # Distribute roughly equally with some overlap for federated learning
            total_images = len(image_files)
            images_per_partition = total_images // num_partitions
            
            # Add some overlap (20% extra) to simulate real federated scenarios
            overlap = int(images_per_partition * 0.2)
            
            # Calculate start and end indices for this partition
            start_idx = (i - 1) * images_per_partition
            end_idx = min(start_idx + images_per_partition + overlap, total_images)
            
            # If it's the last partition, include all remaining images
            if i == num_partitions:
                end_idx = total_images
            
            partition_images = image_files[start_idx:end_idx]
            
            print(f"  Digit {digit_name}: copying {len(partition_images)} images to train{i}")
            
            # Copy images to partition
            for img_file in partition_images:
                dest_file = partition_digit_dir / img_file.name
                shutil.copy2(img_file, dest_file)
    
    print("\nDataset partitioning complete!")
    print(f"Created {num_partitions} training partitions in 'training_data' directory")
    
    # Print summary
    for i in range(1, num_partitions + 1):
        partition_dir = training_data_dir / f"train{i}"
        total_images = sum(len(list(digit_dir.iterdir())) for digit_dir in partition_dir.iterdir())
        print(f"train{i}: {total_images} images")

if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    create_federated_dataset()
