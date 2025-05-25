import * as tf from '@tensorflow/tfjs';
import { loadTensorFlowJSModel, createModelBundle } from './modelConverter';

interface TrainingParams {
  epochs: number;
  batchSize: number;
  learningRate: number;
  activationFunction: string;
  dropoutRate: number;
  numLayers: number;
  unitsPerLayer: number;
}

interface TrainModelParams {
  trainingDataDir: FileList;
  trainingParams: TrainingParams;
  existingModel?: ArrayBuffer | null;
  onProgress?: (progress: number) => void;
}

interface TrainModelResult {
  modelBlob: Blob;
  metadata: {
    accuracy: number;
    loss: number;
    epochs: number;
    batchSize: number;
    trainingTime: number;
  };
}

// Default MNIST-like model architecture
// ...existing code...

// Default MNIST-like model architecture
const createModel = (numClasses: number, params: TrainingParams): tf.Sequential => {
  const model = tf.sequential();
  
  // Input layer - accept 28x28x1 images directly (no flatten layer)
  // The backend test script will feed images in this format
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: params.activationFunction as any
  }));
  
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: params.activationFunction as any
  }));
  
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  
  // Flatten before dense layers
  model.add(tf.layers.flatten());
  
  // Hidden layers
  for (let i = 0; i < params.numLayers; i++) {
    model.add(tf.layers.dense({
      units: params.unitsPerLayer,
      activation: params.activationFunction as any
    }));
    
    if (params.dropoutRate > 0) {
      model.add(tf.layers.dropout({ rate: params.dropoutRate }));
    }
  }
  
  // Output layer
  model.add(tf.layers.dense({
    units: numClasses,
    activation: 'softmax'
  }));
  
  model.compile({
    optimizer: tf.train.adam(params.learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
};

// ...existing code...

const preprocessImage = async (file: File): Promise<tf.Tensor> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      
      // Resize to 28x28 for MNIST compatibility
      canvas.width = 28;
      canvas.height = 28;
      ctx.drawImage(img, 0, 0, 28, 28);
      
      const imageData = ctx.getImageData(0, 0, 28, 28);
      const tensor = tf.browser.fromPixels(imageData, 1)
        .div(255.0); // Normalize to [0, 1]
        // Remove expandDims(0) here - we'll batch them later
      
      resolve(tensor);
    };
    img.src = URL.createObjectURL(file);
  });
};

const loadDataFromDirectory = async (
  files: FileList,
  onProgress?: (progress: number) => void
): Promise<{ xs: tf.Tensor; ys: tf.Tensor; numClasses: number }> => {
  const labelMap = new Map<string, number>();
  const imageFiles: File[] = [];
  const labels: number[] = [];
  
  // Extract labels from file paths and create label mapping
  let labelIndex = 0;
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (file.type.startsWith('image/')) {
      const pathParts = file.webkitRelativePath.split('/');
      const label = pathParts[pathParts.length - 2]; // Parent directory name
      
      if (!labelMap.has(label)) {
        labelMap.set(label, labelIndex++);
      }
      
      imageFiles.push(file);
      labels.push(labelMap.get(label)!);
    }
  }
  
  console.log(`Found ${imageFiles.length} images across ${labelMap.size} classes`);
  
  // Limit dataset size to prevent memory issues
  const maxImages = 5000; // Reduce from potentially 60k images
  if (imageFiles.length > maxImages) {
    console.log(`Limiting dataset to ${maxImages} images for memory efficiency`);
    const indices = Array.from({length: imageFiles.length}, (_, i) => i);
    // Shuffle and take first maxImages
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    const selectedIndices = indices.slice(0, maxImages);
    const selectedFiles = selectedIndices.map(i => imageFiles[i]);
    const selectedLabels = selectedIndices.map(i => labels[i]);
    
    imageFiles.splice(0, imageFiles.length, ...selectedFiles);
    labels.splice(0, labels.length, ...selectedLabels);
  }
  
  // Process images in smaller batches to prevent memory issues
  const batchSize = 100;
  const allTensors: tf.Tensor[] = [];
  
  for (let batchStart = 0; batchStart < imageFiles.length; batchStart += batchSize) {
    const batchEnd = Math.min(batchStart + batchSize, imageFiles.length);
    const batchFiles = imageFiles.slice(batchStart, batchEnd);
    
    console.log(`Processing batch ${Math.floor(batchStart/batchSize) + 1}/${Math.ceil(imageFiles.length/batchSize)}`);
    
    const batchTensors = await Promise.all(
      batchFiles.map(async (file, index) => {
        try {
          const tensor = await preprocessImage(file);
          if (onProgress) {
            const progress = ((batchStart + index) / imageFiles.length) * 30; // 30% for data loading
            onProgress(progress);
          }
          return tensor;
        } catch (error) {
          console.warn(`Failed to process image ${file.name}:`, error);
          return null;
        }
      })
    );
    
    // Filter out failed images and add to collection
    const validTensors = batchTensors.filter(t => t !== null) as tf.Tensor[];
    allTensors.push(...validTensors);
    
    // Force garbage collection between batches
    if (typeof window !== 'undefined' && (window as any).gc) {
      (window as any).gc();
    }
  }
  
  if (allTensors.length === 0) {
    throw new Error('No valid images could be processed');
  }
  
  console.log(`Successfully processed ${allTensors.length} images`);
  
  // Convert to tensors with memory management
  try {
    // Stack tensors without adding extra dimension
    const xs = tf.stack(allTensors); // This creates shape [N, 28, 28, 1]
    const validLabels = labels.slice(0, allTensors.length);
    const ys = tf.oneHot(tf.tensor1d(validLabels, 'int32'), labelMap.size);
    
    // Clean up intermediate tensors
    allTensors.forEach(tensor => tensor.dispose());
    
    return { xs, ys, numClasses: labelMap.size };
  } catch (error) {
    // Clean up on error
    allTensors.forEach(tensor => tensor.dispose());
    throw error;
  }
};

export const trainModel = async (params: TrainModelParams): Promise<TrainModelResult> => {
  const startTime = Date.now();
  
  try {
    // Enable memory debugging
    console.log('Initial memory usage:', tf.memory());
    
    // Load and preprocess data
    params.onProgress?.(0);
    const { xs, ys, numClasses } = await loadDataFromDirectory(
      params.trainingDataDir,
      (progress) => params.onProgress?.(progress)
    );
    
    console.log('Memory after data loading:', tf.memory());
    console.log(`Dataset shape: xs=${xs.shape}, ys=${ys.shape}, classes=${numClasses}`);
    
    // Create or load model
    let model: tf.LayersModel;
    
    if (params.existingModel) {
      try {
        // Load existing TensorFlow.js model
        const loadedModel = await loadTensorFlowJSModel(params.existingModel);
        if (loadedModel) {
          model = loadedModel;
          console.log('Loaded existing TensorFlow.js model');
        } else {
          model = createModel(numClasses, params.trainingParams);
          console.log('Created new model (existing model incompatible)');
        }
      } catch (error) {
        console.warn('Failed to load existing model, creating new one:', error);
        model = createModel(numClasses, params.trainingParams);
      }
    } else {
      // Create new model
      model = createModel(numClasses, params.trainingParams);
    }

    params.onProgress?.(35);
    console.log('Memory after model creation:', tf.memory());
    
    // Reduce batch size for training to prevent memory issues
    const trainingBatchSize = Math.min(params.trainingParams.batchSize, 16);
    console.log(`Training with batch size: ${trainingBatchSize}`);
    
    // Train the model with better error handling
    const history = await model.fit(xs, ys, {
      epochs: params.trainingParams.epochs,
      batchSize: trainingBatchSize,
      validationSplit: 0.15, // Reduced validation split
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const progress = 35 + ((epoch + 1) / params.trainingParams.epochs) * 55;
          params.onProgress?.(progress);
          console.log(`Epoch ${epoch + 1}/${params.trainingParams.epochs}: loss=${logs?.loss?.toFixed(4)}, acc=${logs?.acc?.toFixed(4)}`);
          console.log('Memory during training:', tf.memory());
        },
        onBatchEnd: (batch, logs) => {
          // Periodic memory cleanup during training
          if (batch % 10 === 0 && typeof window !== 'undefined' && (window as any).gc) {
            (window as any).gc();
          }
        }
      }
    });
    
    params.onProgress?.(95);
    console.log('Memory after training:', tf.memory());
    
    // Save model using the new format
    const { modelZip } = await createModelBundle(model);
    
    // Get final metrics with better error handling
    const finalLoss = history.history.loss[history.history.loss.length - 1] as number;
    let finalAccuracy = 0;
    
    if (history.history.acc) {
      finalAccuracy = history.history.acc[history.history.acc.length - 1] as number;
    } else if (history.history.accuracy) {
      finalAccuracy = history.history.accuracy[history.history.accuracy.length - 1] as number;
    }
    
    params.onProgress?.(100);
    
    // Clean up with proper error handling
    try {
      xs.dispose();
      ys.dispose();
      model.dispose();
    } catch (error) {
      console.warn('Error during cleanup:', error);
    }
    
    console.log('Final memory usage:', tf.memory());
    
    const trainingTime = Date.now() - startTime;
    
    return {
      modelBlob: modelZip, // Return the ZIP blob containing both model.json and weights.bin
      metadata: {
        accuracy: finalAccuracy,
        loss: finalLoss,
        epochs: params.trainingParams.epochs,
        batchSize: trainingBatchSize,
        trainingTime
      }
    };
    
  } catch (error) {
    console.error('Training failed:', error);
    console.log('Memory on error:', tf.memory());
    
    // Force cleanup on error
    try {
      tf.disposeVariables();
    } catch (cleanupError) {
      console.warn('Error during cleanup:', cleanupError);
    }
    
    throw error;
  }
};
