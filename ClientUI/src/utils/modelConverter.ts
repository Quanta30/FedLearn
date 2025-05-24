import * as tf from '@tensorflow/tfjs';

export const loadTensorFlowJSModel = async (modelData: ArrayBuffer): Promise<tf.LayersModel | null> => {
  try {
    // Check if the data is HDF5 format (starts with PK)
    const uint8Array = new Uint8Array(modelData);
    const header = String.fromCharCode(...uint8Array.slice(0, 2));
    
    if (header === 'PK') {
      console.warn('Model is in HDF5 format, cannot load with TensorFlow.js');
      return null;
    }
    
    // Convert ArrayBuffer back to JSON string
    const decoder = new TextDecoder();
    const jsonString = decoder.decode(modelData);
    
    // Try to parse as JSON
    let modelConfig;
    try {
      modelConfig = JSON.parse(jsonString);
    } catch (parseError) {
      console.warn('Model data is not valid JSON format');
      return null;
    }
    
    // Validate that this is a TensorFlow.js model
    if (!modelConfig.modelTopology || !modelConfig.weightsManifest) {
      console.warn('Invalid TensorFlow.js model format');
      return null;
    }
    
    // Create a temporary model from the topology
    const model = await tf.loadLayersModel(tf.io.fromMemory(modelConfig));
    
    console.log('Successfully loaded existing TensorFlow.js model');
    return model;
  } catch (error) {
    console.warn('Failed to load TensorFlow.js model:', error);
    return null;
  }
};

export const createModelBundle = async (model: tf.LayersModel): Promise<{ modelZip: Blob }> => {
  return new Promise((resolve, reject) => {
    model.save(tf.io.withSaveHandler(async (artifacts) => {
      try {
        // Create proper TensorFlow.js format
        const modelJson = {
          modelTopology: artifacts.modelTopology,
          weightsManifest: [{
            paths: ['weights.bin'],
            weights: artifacts.weightSpecs
          }],
          format: artifacts.format,
          generatedBy: artifacts.generatedBy,
          convertedBy: artifacts.convertedBy
        };
        
        // Create individual files
        const modelJsonData = JSON.stringify(modelJson, null, 2);
        const weightsData = artifacts.weightData as ArrayBuffer;
        
        // Create a zip file using JSZip
        const JSZip = (await import('jszip')).default;
        const zip = new JSZip();
        
        // Add files to zip
        zip.file('model.json', modelJsonData);
        zip.file('weights.bin', weightsData);
        
        // Generate zip blob
        const zipBlob = await zip.generateAsync({ type: 'blob' });
        
        resolve({ modelZip: zipBlob });
        return { modelArtifactsInfo: { dateSaved: new Date() } };
      } catch (error) {
        reject(error);
      }
    }));
  });
};
