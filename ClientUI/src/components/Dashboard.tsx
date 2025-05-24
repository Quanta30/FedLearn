import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FaUser, FaHistory, FaDownload, FaCog, FaLock, FaUserCog, FaChartLine, FaUsers, FaClipboardList, FaExclamationCircle, FaArrowLeft, FaPlus, FaEdit } from 'react-icons/fa';
import CONFIG from '../config/config';
import JSZip from 'jszip';

interface Contributor {
  username: string;
  profileLink: string;
  isOwner?: boolean;
}

interface DashboardProps {
  token: string;
  currentProject: Project;
  setCurrentProject: (project: Project | null) => void;
  setContribScreen: (value: boolean) => void;
}

const Dashboard: React.FC<DashboardProps> = ({ 
  token, 
  currentProject, 
  setCurrentProject, 
  setContribScreen 
}) => {
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
        }
      } catch (error) {
        console.error('Error fetching project data:', error);
      }
    };

    fetchProjects();
  }, [currentProject, token]);

  const handleSearch = async () => {
    try {
      const response = await axios.get(`${CONFIG.BACKEND_URI}/project/search/${searchQuery}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setSearchResults(response.data.usernames.map(username => ({ username, profileLink: `/users/${username}` })));
    } catch (error) {
      console.error('Error searching for users:', error);
    }
  };
  const handleDownloadModel = async () => {
    try {
      const response = await axios.get(`${CONFIG.BACKEND_URI}/project/${projectName}/model`, {
        headers: { Authorization: `Bearer ${token}` },
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'model.zip');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading model:', error);
      alert('Error downloading model');
    }
  };
  const handleAddContributor = async (username: string) => {
    try {
      await axios.post(`${CONFIG.BACKEND_URI}/project/${projectName}/collaborators`, { username, projectName }, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setIsAddContributorModalOpen(false);
      setSearchQuery('');
      setSearchResults([]);
      setNewContributor('');
      // Refresh project data
      const response = await axios.get(`${CONFIG.BACKEND_URI}/project`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const projects = response.data.projects || [];
      const projectData = projects.find(project => project.uuid === currentProject);
      if (projectData) {
        setCurrentContributors([
          {
            username: projectData.owner.username || 'Unknown User',
            profileLink: `/users/${projectData.owner._id || '#'}`,
            isOwner: true
          },
          ...(projectData.collaborators || []).map(collaborator => ({
            username: collaborator.username || 'Unknown User',
            profileLink: `/users/${collaborator._id || '#'}`
          }))
        ]);
      }
    } catch (error) {
      console.error('Error adding contributor:', error);
    }
  };

  const handleTestModel = async () => {
    try {
      const response = await axios.post(`${CONFIG.BACKEND_URI}/project/${projectName}/test-model`, { temp : "TEMP" }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      alert(`Model tested successfully! Accuracy: ${response.data.accuracy}`);
    } catch (error) {
      console.error('Error testing model:', error);
      alert('Error testing model');
    }
  };

  const handleEditProject = async () => {
    try {
      await axios.put(`${CONFIG.BACKEND_URI}/project/${currentProject}`, editProjectDetails, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setIsEditProjectModalOpen(false);
      // Refresh project data
      const response = await axios.get(`${CONFIG.BACKEND_URI}/project`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const projects = response.data.projects || [];
      const projectData = projects.find(project => project.uuid === currentProject);
      if (projectData) {
        setProjectName(projectData.name || 'No Project Name');
        setDescription(projectData.description || 'No Description Available');
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
      }
    } catch (error) {
      console.error('Error updating project:', error);
    }
  };

  const handleAddTestFiles = async () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.directory = true;
    input.multiple = true;
    input.onchange = async (e: any) => {
      const files = e.target.files;
      const formData = new FormData();

  
      const zip = new JSZip();
      const folder = zip.folder('test_set');
      for (let i = 0; i < files.length; i++) {
        const relativePath = files[i].webkitRelativePath.replace(/^test_set\//, '');
        folder.file(relativePath, files[i]);
      }
      const content = await zip.generateAsync({ type: 'blob' });
      formData.append('testSetZip', content, 'test_set.zip');
  
      try {
        await axios.post(`${CONFIG.BACKEND_URI}/project/${projectName}/test-file`, formData, {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
          },
        });
        alert('Test files uploaded successfully!');
      } catch (error) {
        console.error('Error uploading test files:', error);
        alert('Error uploading test files');
      }
    };
    input.click();
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Navigation Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <button
            onClick={() => setCurrentProject(null)}
            className="mr-4 px-3 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
          >
            ← Back to Projects
          </button>
          <h1 className="text-3xl font-bold text-gray-800">Project Dashboard</h1>
        </div>
        <button
          onClick={() => setContribScreen(true)}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors font-medium"
        >
          Contribute to Project
        </button>
      </div>

      <div className="navbar bg-base-100 rounded-box shadow-lg mb-4">
        <div className="flex-1">
          <h1 className="text-2xl font-bold px-4">{projectName}</h1>
        </div>
        <div className="flex-none">
          <button className="btn btn-ghost btn-circle" onClick={() => setCurrentProject('')}>
            <FaArrowLeft className="h-5 w-5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Project Overview */}
        <div className="card bg-base-100 shadow-xl">
          <div className="card-body">
            <h2 className="card-title">Project Overview</h2>
            <div className="stats stats-vertical shadow">
              <div className="stat">
                <div className="stat-figure text-primary">
                  <FaUsers className="inline-block w-8 h-8 stroke-current" />
                </div>
                <div className="stat-title">Collaborators</div>
                <div className="stat-value">{collaboratorsCount}</div>
              </div>
              <div className="stat">
                <div className="stat-figure text-secondary">
                  <FaChartLine className="inline-block w-8 h-8 stroke-current" />
                </div>
                <div className="stat-title">Model Accuracy</div>
                <div className="stat-value">{modelAccuracy}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Project Description */}
        <div className="card bg-base-100 shadow-xl">
  <div className="card-body">
    <h2 className="card-title">Project Description</h2>
    <p>{description}</p>
    <div className="card-actions justify-end">
      <button className="btn btn-primary" onClick={() => setContribScreen(true)} >Contribute Now</button>
      <button className="btn btn-secondary" onClick={handleAddTestFiles}>
        Add Test Files
      </button>
      <button className="btn btn-secondary" onClick={handleDownloadModel}>
        <FaDownload className="mr-2" />
        Download Model
      </button>
    </div>
  </div>
</div>

        {/* Current Contributors */}
        <div className="card bg-base-100 shadow-xl">
          <div className="card-body">
            <h2 className="card-title">Current Contributors</h2>
            <ul className="menu bg-base-200 rounded-box">
              {currentContributors.length > 0 ? (
                currentContributors.map((contributor, index) => (
                  <li key={index}>
                    <a href={contributor.profileLink} className="flex items-center">
                      <FaUser className="mr-2" />
                      {contributor.username} {contributor.isOwner && '(Owner)'}
                    </a>
                  </li>
                ))
              ) : (
                <li className="flex items-center">
                  <FaExclamationCircle className="mr-2" />
                  No Contributors Available
                </li>
              )}
            </ul>
            <button className="btn btn-primary mt-4" onClick={() => setIsAddContributorModalOpen(true)}>
              <FaPlus className="mr-2" />
              Add Contributor
            </button>
            {currentContributors.some(contributor => contributor.isOwner) && (
              <button className="btn btn-secondary mt-4" onClick={() => setIsEditProjectModalOpen(true)}>
                <FaEdit className="mr-2" />
                Edit Project
              </button>
            )}
          </div>
        </div>
<div className="card bg-base-100 shadow-xl">
  <div className="card-body">
    <h2 className="card-title">Test Model</h2>
    <div className="card-actions justify-end">
      <button className="btn btn-primary" onClick={handleTestModel}>
        Test Now
      </button>
    </div>
  </div>
</div>
        {/* History */}
        <div className="card bg-base-100 shadow-xl md:col-span-2">
          <div className="card-body">
            <h2 className="card-title">
              <FaHistory className="mr-2" />
              History
            </h2>
            <ul className="menu bg-base-200 rounded-box">
              {history.length > 0 ? (
                history.map((contributor, index) => (
                  <li key={index}>
                    <a href={contributor.profileLink} className="flex items-center">
                      <FaUser className="mr-2" />
                      {contributor.username}
                    </a>
                  </li>
                ))
              ) : (
                <li className="flex items-center">
                  <FaExclamationCircle className="mr-2" />
                  No History Available
                </li>
              )}
            </ul>
          </div>
        </div>
      </div>

      {/* Add Contributor Modal */}
      {isAddContributorModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-base-100 p-6 rounded-lg shadow-xl w-96">
            <h3 className="text-2xl font-semibold mb-4">Add Contributor</h3>
            <div className="mb-4">
              <label className="block mb-2">Search Username</label>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input input-bordered w-full"
                placeholder="Search Username"
              />
              <button className="btn btn-primary mt-2" onClick={handleSearch}>
                Search
              </button>
            </div>
            <ul className="menu bg-base-200 rounded-box">
              {searchResults.length > 0 ? (
                searchResults.map((result, index) => (
                  <li key={index}>
                    <a className="flex items-center" onClick={() => handleAddContributor(result.username)}>
                      <FaUser className="mr-2" />
                      {result.username}
                    </a>
                  </li>
                ))
              ) : (
                <li className="flex items-center">
                  <FaExclamationCircle className="mr-2" />
                  No Users Found
                </li>
              )}
            </ul>
            <div className="flex justify-end mt-4">
              <button className="btn btn-secondary mr-2" onClick={() => setIsAddContributorModalOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Project Modal */}
      {isEditProjectModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-base-100 p-6 rounded-lg shadow-xl w-96">
            <h3 className="text-2xl font-semibold mb-4">Edit Project</h3>
            <div className="mb-4">
              <label className="block mb-2">Project Name</label>
              <input
                type="text"
                name="name"
                value={editProjectDetails.name}
                onChange={(e) => setEditProjectDetails({ ...editProjectDetails, name: e.target.value })}
                className="input input-bordered w-full"
                placeholder="Project Name"
              />
            </div>
            <div className="mb-4">
              <label className="block mb-2">Description</label>
              <textarea
                name="description"
                value={editProjectDetails.description}
                onChange={(e) => setEditProjectDetails({ ...editProjectDetails, description: e.target.value })}
                className="input input-bordered w-full"
                placeholder="Project Description"
              />
            </div>
            <div className="mb-4">
              <label className="block mb-2">Activation Function</label>
              <input
                type="text"
                name="activation_function"
                value={editProjectDetails.activation_function}
                onChange={(e) => setEditProjectDetails({ ...editProjectDetails, activation_function: e.target.value })}
                className="input input-bordered w-full"
                placeholder="Activation Function"
              />
            </div>
            <div className="mb-4">
              <label className="block mb-2">Dropout Rate</label>
              <input
                type="text"
                name="dropout_rate"
                value={editProjectDetails.dropout_rate}
                onChange={(e) => setEditProjectDetails({ ...editProjectDetails, dropout_rate: e.target.value })}
                className="input input-bordered w-full"
                placeholder="Dropout Rate"
              />
            </div>
            <div className="mb-4">
              <label className="block mb-2">Combining Method</label>
              <input
                type="text"
                name="combining_method"
                value={editProjectDetails.combining_method}
                onChange={(e) => setEditProjectDetails({ ...editProjectDetails, combining_method: e.target.value })}
                className="input input-bordered w-full"
                placeholder="Combining Method"
              />
            </div>
            <div className="mb-4">
              <label className="block mb-2">Input Shape</label>
              <input
                type="text"
                name="input_shape"
                value={editProjectDetails.input_shape}
                onChange={(e) => setEditProjectDetails({ ...editProjectDetails, input_shape: e.target.value })}
                className="input input-bordered w-full"
                placeholder="Input Shape"
              />
            </div>
            <div className="mb-4">
              <label className="block mb-2">Number of Layers</label>
              <input
                type="text"
                name="num_layers"
                value={editProjectDetails.num_layers}
                onChange={(e) => setEditProjectDetails({ ...editProjectDetails, num_layers: e.target.value })}
                className="input input-bordered w-full"
                placeholder="Number of Layers"
              />
            </div>
            <div className="mb-4">
              <label className="block mb-2">Units per Layer</label>
              <input
                type="text"
                name="units_per_layer"
                value={editProjectDetails.units_per_layer}
                onChange={(e) => setEditProjectDetails({ ...editProjectDetails, units_per_layer: e.target.value })}
                className="input input-bordered w-full"
                placeholder="Units per Layer"
              />
            </div>
            <div className="flex justify-end mt-4">
              <button className="btn btn-secondary mr-2" onClick={() => setIsEditProjectModalOpen(false)}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleEditProject}>
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;