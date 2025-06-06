// Contrib.tsx
import React, { useState, ChangeEvent, FormEvent, useEffect } from 'react';
import axios from 'axios';
import CONFIG from '../config/config';
import { trainModel } from '../utils/trainModel';

interface ContribProps {
  token: string;
  currentProject: {
    name: string;
  };
  setContribScreen: (value: boolean) => void;
}

interface TrainingParams {
  epochs: number;
  batchSize: number;
  learningRate: number;
  activationFunction: string;
  dropoutRate: number;
  numLayers: number;
  unitsPerLayer: number;
}

const Contrib: React.FC<ContribProps> = ({ token, currentProject, setContribScreen }) => {
  const [trainingDataDir, setTrainingDataDir] = useState<FileList | null>(null);
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    activationFunction: 'relu',
    dropoutRate: 0.2,
    numLayers: 3,
    unitsPerLayer: 128
  });
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [trainingProgress, setTrainingProgress] = useState<number>(0);

  const [projectName, setProjectName] = useState('');
  const [collaboratorsCount, setCollaboratorsCount] = useState(0);
  const [modelAccuracy, setModelAccuracy] = useState('');
  const [description, setDescription] = useState('');
  const [history, setHistory] = useState<Contributor[]>([]);
  const [currentContributors, setCurrentContributors] = useState<Contributor[]>([]);
  const [isAddContributorModalOpen, setIsAddContributorModalOpen] = useState(false);
  const [isEditProjectModalOpen, setIsEditProjectModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Contributor[]>([]);
  const [newContributor, setNewContributor] = useState('');
  const [editProjectDetails, setEditProjectDetails] = useState({
    name: '',
    description: '',
    activation_function: '',
    dropout_rate: '',
    combining_method: '',
    input_shape: '',
    num_layers: '',
    units_per_layer: ''
  });


  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const response = await axios.get(`${CONFIG.BACKEND_URI}/project`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const projects = response.data.projects || [];
        const projectData = projects.find(project => project.uuid === currentProject);

        if (projectData) {
          setProjectName(projectData.name || 'No Project Name');
          setCollaboratorsCount((projectData.collaborators && projectData.collaborators.length) || 0);
          setModelAccuracy(projectData.accuracy || 'N/A');
          setDescription(projectData.description || 'No Description Available');
          
          // Add null checks for contributions and user objects
          setHistory((projectData.contributions || []).map(contribution => ({
            username: (contribution.user && contribution.user.username) || 'Unknown User',
            profileLink: `/users/${(contribution.user && contribution.user._id) || '#'}`
          })).filter(item => item.username !== 'Unknown User')); // Filter out invalid entries
          
          // Add null checks for owner and collaborators
          const contributors = [];
          
          // Add owner if exists
          if (projectData.owner && projectData.owner.username) {
            contributors.push({
              username: projectData.owner.username,
              profileLink: `/users/${projectData.owner._id || '#'}`,
              isOwner: true
            });
          }
          
          // Add collaborators if they exist
          if (projectData.collaborators && Array.isArray(projectData.collaborators)) {
            const validCollaborators = projectData.collaborators
              .filter(collaborator => collaborator && collaborator.username)
              .map(collaborator => ({
                username: collaborator.username,
                profileLink: `/users/${collaborator._id || '#'}`
              }));
            contributors.push(...validCollaborators);
          }
          
          setCurrentContributors(contributors);
          setEditProjectDetails({
            name: projectData.name,
            description: projectData.description,
            activation_function: projectData.activation_function,
            dropout_rate: projectData.dropout_rate,
            combining_method: projectData.combining_method,
            input_shape: projectData.input_shape,
            num_layers: projectData.num_layers,
            units_per_layer: projectData.units_per_layer
          });
          
          // Set training parameters from project config or use MNIST defaults
          setTrainingParams({
            epochs: projectData.epochs || 10,
            batchSize: projectData.batch_size || 32,
            learningRate: projectData.learning_rate || 0.001,
            activationFunction: projectData.activation_function || 'relu',
            dropoutRate: parseFloat(projectData.dropout_rate) || 0.2,
            numLayers: parseInt(projectData.num_layers) || 3,
            unitsPerLayer: parseInt(projectData.units_per_layer) || 128
          });
        }
      } catch (error) {
        console.error('Error fetching project data:', error);
      }
    };

    fetchProjects();
  }, [currentProject, token]);

  const handleDirectoryChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setTrainingDataDir(e.target.files);
    }
  };

  const handleParamChange = (param: keyof TrainingParams, value: string | number) => {
    setTrainingParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const downloadExistingModel = async (): Promise<ArrayBuffer | null> => {
    try {
      setStatusMessage('Downloading existing model...');
      console.log('Downloading existing model...');
      
      const response = await axios.get(
        `${CONFIG.BACKEND_URI}/project/${projectName}/model`,
        {
          headers: { Authorization: `Bearer ${token}` },
          responseType: 'arraybuffer' // Keep as arraybuffer since backend serves HDF5
        }
      );
      
      // Check if the response is HDF5 format (starts with PK) or JSON
      const uint8Array = new Uint8Array(response.data);
      const header = String.fromCharCode(...uint8Array.slice(0, 2));
      
      if (header === 'PK') {
        console.log('Downloaded model is in HDF5 format, skipping for now');
        return null; // Skip HDF5 models for now
      } else {
        console.log('Model downloaded successfully');
        return response.data;
      }
    } catch (error: any) {
      console.log('No existing model found, starting fresh training');
      return null;
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!trainingDataDir || trainingDataDir.length === 0) {
      alert('Please select a training data directory.');
      return;
    }

    setIsSubmitting(true);
    setTrainingProgress(0);
    setStatusMessage('Starting training process...');
    console.log('Starting training process...');

    try {
      // Download existing model if available
      const existingModel = await downloadExistingModel();
      
      // Train the model locally
      setStatusMessage('Training model locally...');
      const { modelBlob, metadata } = await trainModel({
        trainingDataDir,
        trainingParams,
        existingModel,
        onProgress: (progress: number) => {
          setTrainingProgress(progress);
          setStatusMessage(`Training progress: ${Math.round(progress)}%`);
        }
      });

      // Calculate SHA1 hash of the model
      setStatusMessage('Preparing model for upload...');
      const arrayBuffer = await modelBlob.arrayBuffer();
      const hashBuffer = await crypto.subtle.digest('SHA-1', arrayBuffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const sha1Hash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

      // Upload the trained model
      setStatusMessage('Uploading trained model...');
      const formData = new FormData();
      formData.append('file', modelBlob, `model_${Date.now()}.zip`); // Changed to .zip extension
      formData.append('sha1', sha1Hash);
      formData.append('metadata', JSON.stringify(metadata));
      formData.append('format', 'tensorflowjs'); // Add format identifier

      const response = await axios.post(
        `${CONFIG.BACKEND_URI}/project/${projectName}/contribute`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${token}`
          },
        }
      );

      setStatusMessage('Contribution successful!');
      console.log('Response:', response.data);
      setTrainingDataDir(null);
      setTrainingProgress(0);
      
      // Reset file input
      const fileInput = document.getElementById('trainingData') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
      
    } catch (error: any) {
      console.error('Error during contribution:', error);
      
      // More detailed error message
      if (error.message.includes('dimension')) {
        setStatusMessage('Tensor shape error during training. Please check your training data format.');
      } else if (error.message.includes('memory')) {
        setStatusMessage('Memory error during training. Try reducing batch size or dataset size.');
      } else {
        setStatusMessage('An error occurred during contribution.');
      }
      
      alert(`Error during contribution: ${error.message}\nPlease check the console for details.`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-8 p-6 bg-white shadow-md rounded">
      {/* Navigation Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <button
            onClick={() => setContribScreen(false)}
            className="mr-4 px-3 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
          >
            ← Back to Dashboard
          </button>
          <h2 className="text-2xl font-semibold">Contribute to Project</h2>
        </div>
      </div>

      {/* Project Info */}
      <div className="mb-4 p-4 bg-blue-50 rounded">
        <p className="text-lg font-medium text-blue-800">
          Contributing to: <span className="font-bold">{projectName}</span>
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Training Data Directory Input */}
        <div className="mb-4">
          <label className="block text-gray-700 mb-2" htmlFor="trainingData">
            Select Training Data Directory
          </label>
          <input
            type="file"
            id="trainingData"
            webkitdirectory=""
            directory=""
            multiple
            onChange={handleDirectoryChange}
            disabled={isSubmitting}
            className="w-full px-3 py-2 border rounded"
            required
          />
          <small className="text-gray-500">
            Select a directory containing labeled folders with training images
          </small>
        </div>

        {/* Training Parameters */}
        <div className="mb-4 p-4 border rounded bg-gray-50">
          <h3 className="text-lg font-medium mb-3">Training Parameters</h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-700 mb-1">Epochs</label>
              <input
                type="number"
                min="1"
                value={trainingParams.epochs}
                onChange={(e) => handleParamChange('epochs', parseInt(e.target.value))}
                disabled={isSubmitting}
                className="w-full px-3 py-2 border rounded"
              />
            </div>
            
            <div>
              <label className="block text-gray-700 mb-1">Batch Size</label>
              <input
                type="number"
                min="1"
                value={trainingParams.batchSize}
                onChange={(e) => handleParamChange('batchSize', parseInt(e.target.value))}
                disabled={isSubmitting}
                className="w-full px-3 py-2 border rounded"
              />
            </div>
            
            <div>
              <label className="block text-gray-700 mb-1">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                min="0.0001"
                max="1"
                value={trainingParams.learningRate}
                onChange={(e) => handleParamChange('learningRate', parseFloat(e.target.value))}
                disabled={isSubmitting}
                className="w-full px-3 py-2 border rounded"
              />
            </div>
            
            <div>
              <label className="block text-gray-700 mb-1">Dropout Rate</label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={trainingParams.dropoutRate}
                onChange={(e) => handleParamChange('dropoutRate', parseFloat(e.target.value))}
                disabled={isSubmitting}
                className="w-full px-3 py-2 border rounded"
              />
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        {isSubmitting && trainingProgress > 0 && (
          <div className="mb-4">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-600 mt-1">Training Progress: {Math.round(trainingProgress)}%</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isSubmitting}
          className={`w-full py-2 px-4 bg-blue-500 text-white rounded hover:bg-blue-600 ${
            isSubmitting ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {isSubmitting ? 'Training & Contributing...' : 'Train & Contribute'}
        </button>
      </form>

      {/* Status Message */}
      {statusMessage && (
        <div className="mt-4">
          <p className="text-center text-gray-700">{statusMessage}</p>
        </div>
      )}
    </div>
  );
};

export default Contrib;