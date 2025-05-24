import * as tf from '@tensorflow/tfjs';

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
const createModel = (numClasses: number, params: TrainingParams): tf.Sequential => {
  const model = tf.sequential();
  
  // Input layer - assuming 28x28 grayscale images for MNIST
  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  
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
        .div(255.0) // Normalize to [0, 1]
        .expandDims(0);
      
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
  const images: tf.Tensor[] = [];
  const labels: number[] = [];
  
  // Extract labels from file paths and create label mapping
  let labelIndex = 0;
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const pathParts = file.webkitRelativePath.split('/');
    const label = pathParts[pathParts.length - 2]; // Parent directory name
    
    if (!labelMap.has(label)) {
      labelMap.set(label, labelIndex++);
    }
  }
  
  // Process images
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (file.type.startsWith('image/')) {
      const pathParts = file.webkitRelativePath.split('/');
      const label = pathParts[pathParts.length - 2];
      const labelIndex = labelMap.get(label)!;
      
      try {
        const tensor = await preprocessImage(file);
        images.push(tensor);
        labels.push(labelIndex);
        
        if (onProgress) {
          onProgress((i / files.length) * 50); // First 50% for data loading
        }
      } catch (error) {
        console.warn(`Failed to process image ${file.name}:`, error);
      }
    }
  }
  
  // Convert to tensors
  const xs = tf.stack(images);
  const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), labelMap.size);
  
  // Clean up intermediate tensors
  images.forEach(tensor => tensor.dispose());
  
  return { xs, ys, numClasses: labelMap.size };
};

export const trainModel = async (params: TrainModelParams): Promise<TrainModelResult> => {
  const startTime = Date.now();
  
  try {
    // Load and preprocess data
    params.onProgress?.(0);
    const { xs, ys, numClasses } = await loadDataFromDirectory(
      params.trainingDataDir,
      (progress) => params.onProgress?.(progress * 0.3) // 30% for data loading
    );
    
    // Create or load model
    let model: tf.LayersModel;
    
    if (params.existingModel) {
      // Load existing model from ArrayBuffer
      const modelBlob = new Blob([params.existingModel]);
      model = await tf.loadLayersModel(tf.io.browserFiles([modelBlob]));
    } else {
      // Create new model
      model = createModel(numClasses, params.trainingParams);
    }
    
    params.onProgress?.(40);
    
    // Train the model
    const history = await model.fit(xs, ys, {
      epochs: params.trainingParams.epochs,
      batchSize: params.trainingParams.batchSize,
      validationSplit: 0.2,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const progress = 40 + ((epoch + 1) / params.trainingParams.epochs) * 50;
          params.onProgress?.(progress);
          console.log(`Epoch ${epoch + 1}: loss = ${logs?.loss}, accuracy = ${logs?.acc}`);
        }
      }
    });
    
    params.onProgress?.(95);
    
    // Save model to blob
    const modelBlob = await new Promise<Blob>((resolve) => {
      model.save(tf.io.withSaveHandler(async (artifacts) => {
        // Create proper HDF5-like format for backend compatibility
        const modelTopologyStr = JSON.stringify(artifacts.modelTopology);
        const weightsData = artifacts.weightData as ArrayBuffer;
        
        // Create a proper model file that mimics Keras/HDF5 format
        // This creates a simple binary format: [JSON_LENGTH][JSON][WEIGHTS]
        const jsonBytes = new TextEncoder().encode(modelTopologyStr);
        const jsonLength = new Uint32Array([jsonBytes.length]);
        
        const combinedBuffer = new ArrayBuffer(
          4 + jsonBytes.length + weightsData.byteLength
        );
        
        const view = new Uint8Array(combinedBuffer);
        view.set(new Uint8Array(jsonLength.buffer), 0);
        view.set(jsonBytes, 4);
        view.set(new Uint8Array(weightsData), 4 + jsonBytes.length);
        
        const modelBlob = new Blob([combinedBuffer], { 
          type: 'application/octet-stream' 
        });
        
        resolve(modelBlob);
        return { modelArtifactsInfo: { dateSaved: new Date() } };
      }));
    });
    
    // Get final metrics
    const finalLoss = history.history.loss[history.history.loss.length - 1] as number;
    const finalAccuracy = history.history.acc ? 
      history.history.acc[history.history.acc.length - 1] as number : 
      history.history.accuracy[history.history.accuracy.length - 1] as number;
    
    params.onProgress?.(100);
    
    // Clean up
    xs.dispose();
    ys.dispose();
    model.dispose();
    
    const trainingTime = Date.now() - startTime;
    
    return {
      modelBlob,
      metadata: {
        accuracy: finalAccuracy,
        loss: finalLoss,
        epochs: params.trainingParams.epochs,
        batchSize: params.trainingParams.batchSize,
        trainingTime
      }
    };
    
  } catch (error) {
    console.error('Training failed:', error);
    throw error;
  }
};
