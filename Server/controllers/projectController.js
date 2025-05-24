const Project = require('../models/Project'); 
const User = require('../models/User');
const Contribution = require('../models/Contribution')
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const AdmZip = require('adm-zip');

const mongoose = require('mongoose')
const formidable = require('formidable');



const createProject = async (req, res) => {
    const { name, description, isPrivate, activation_function, dropout_rate, combining_method, input_shape, num_layers, units_per_layer, num_classes } = req.body;
    const owner = req.email;

    try {
        const existingProject = await Project.findOne({ name });
        if (existingProject) {
            return res.status(400).json({ message: 'Project name must be unique.' });
        }

        const user = await User.findOne({ email: owner });
        if (!user) {
            return res.status(404).json({ message: 'User not found.' });
        }

        const newProject = new Project({
            name,
            description,
            owner: user._id,
            isPrivate,
            activation_function,
            dropout_rate,
            combining_method,
            input_shape,
            num_layers,
            units_per_layer,
            num_classes
        });

        // Create project folder structure
        const userFolderPath = path.join(__dirname, '..', 'data', 'users', user.username);
        const projectFolderPath = path.join(userFolderPath, name);
        const pyFolderPath = path.join(projectFolderPath, 'py');

        if (!fs.existsSync(userFolderPath)) {
            fs.mkdirSync(userFolderPath, { recursive: true });
        }

        if (!fs.existsSync(projectFolderPath)) {
            fs.mkdirSync(projectFolderPath, { recursive: true });
        }

        if (!fs.existsSync(pyFolderPath)) {
            fs.mkdirSync(pyFolderPath, { recursive: true });
        }

        // Create config.txt file
        const configContent = `
activation_function=${activation_function}
dropout_rate=${dropout_rate}
combining_method=${combining_method}
input_shape=${input_shape}
num_layers=${num_layers}
units_per_layer=${units_per_layer}
num_classes=${num_classes}
        `;
        fs.writeFileSync(path.join(pyFolderPath, 'config.txt'), configContent.trim());

        // Execute Python code
        const venvActivatePath = path.join(__dirname, '..', 'py', '.venv', 'bin', 'activate');
        const initScriptPath = path.join(__dirname, '..', 'py', 'init.py');
        const command = `cd ${path.join(__dirname, '..', 'py')} && . ${venvActivatePath} && python ${initScriptPath} ${user.username} ${name}`;

        exec(command, async (error, stdout, stderr) => {
            if (error) {
                console.error(`Error executing Python script: ${error.message}`);
                return res.status(500).json({ message: 'Error executing Python script', error: error.message });
            }

            console.log(`Python script output: ${stdout}`);
            console.error(`Python script error output: ${stderr}`);

            // Check if model files were created after init script
            const pyUserPath = path.join(__dirname, '..', 'py', 'users', user.username);
            const pyProjectPath = path.join(pyUserPath, name);
            console.log(`Checking if model files were created in: ${pyProjectPath}`);
            
            if (fs.existsSync(pyProjectPath)) {
                const filesInPyProject = fs.readdirSync(pyProjectPath);
                console.log(`Files created by init script: ${filesInPyProject.join(', ')}`);
            } else {
                console.log(`Python project directory was not created: ${pyProjectPath}`);
            }

            try {
                await newProject.save();
                res.status(201).json({ message: 'Project created successfully!', project: newProject });
            } catch (error) {
                res.status(400).json({ message: 'Error creating project', error });
            }
        });
    } catch (error) {
        res.status(400).json({ message: 'Error creating project', error });
    }
};




const deleteProject = async (req, res) => {
    const { projectId } = req.params;
    const ownerUsername = req.username; 

    try {
        const project = await Project.findById(projectId).populate('owner');
        if (!project) {
            return res.status(404).json({ message: 'Project not found' });
        }

        if (project.owner.username !== ownerUsername) {
            return res.status(403).json({ message: 'You do not have permission to delete this project' });
        }

        await Project.findByIdAndDelete(projectId);
        res.status(200).json({ message: 'Project deleted successfully!' });
    } catch (error) {
        res.status(400).json({ message: 'Error deleting project', error });
    }
};

const getAllProjects = async (req, res) => {
    const ownerEmail = req.email;

    try {
        const user = await User.findOne({ email: ownerEmail });
        if (!user) {
            return res.status(404).json({ message: 'User not found.' });
        }

        const ownedProjects = await Project.find({ owner: user._id })
            .populate('owner')
            .populate('collaborators');

        const collaboratedProjects = await Project.find({ collaborators: user._id })
            .populate('owner')
            .populate('collaborators');

        const allProjects = [...ownedProjects, ...collaboratedProjects].reduce((acc, project) => {
            if (!acc.some(p => p._id.equals(project._id))) {
                acc.push(project);
            }
            return acc;
        }, []);

        res.status(200).json({ projects: allProjects });
    } catch (error) {
        res.status(400).json({ message: 'Error retrieving projects', error });
    }
};

const updateProject = async (req, res) => {
    const { projectId } = req.params;
    const { name, description, isPrivate, activation_function, dropout_rate, combining_method, input_shape, num_layers, units_per_layer } = req.body;
    const ownerEmail = req.email;

    try {
        const project = await Project.findById(projectId);
        if (!project) {
            return res.status(404).json({ message: 'Project not found.' });
        }

        const user = await User.findOne({ email: ownerEmail });
        if (!user || !project.owner.equals(user._id)) {
            return res.status(403).json({ message: 'Only the project owner can update the project.' });
        }

        project.name = name || project.name;
        project.description = description || project.description;
        project.isPrivate = isPrivate !== undefined ? isPrivate : project.isPrivate;
        project.activation_function = activation_function || project.activation_function;
        project.dropout_rate = dropout_rate || project.dropout_rate;
        project.combining_method = combining_method || project.combining_method;
        project.input_shape = input_shape || project.input_shape;
        project.num_layers = num_layers || project.num_layers;
        project.units_per_layer = units_per_layer || project.units_per_layer;

        await project.save();

        // Update config.txt file
        const userFolderPath = path.join(__dirname, '..', 'data', 'users', user.username);
        const projectFolderPath = path.join(userFolderPath, project.name);
        const configContent = `
activation_function=${project.activation_function}
dropout_rate=${project.dropout_rate}
combining_method=${project.combining_method}
input_shape=${project.input_shape}
num_layers=${project.num_layers}
units_per_layer=${project.units_per_layer}
combine_latest=True
latest_model=
        `;
        fs.writeFileSync(path.join(projectFolderPath, 'config.txt'), configContent.trim());

        res.status(200).json({ message: 'Project updated successfully!', project });
    } catch (error) {
        res.status(400).json({ message: 'Error updating project', error });
    }
};



const addCollaborator = async (req, res) => {
    const { username, projectName } = req.body;
    const ownerEmail = req.email;

    try {
        const user = await User.findOne({ email: ownerEmail });
        if (!user) {
            return res.status(404).json({ message: 'User not found.' });
        }

        const project = await Project.findOne({ name: projectName, owner: user._id });
        if (!project) {
            console.log(projectName)
            return res.status(404).json({ message: 'Project not found.' });
        }

        if (!project.owner.equals(user._id)) {
            return res.status(403).json({ message: 'Only the project owner can add collaborators.' });
        }

        const collaborator = await User.findOne({ username });
        if (!collaborator) {
            return res.status(404).json({ message: 'Collaborator not found.' });
        }

        if (project.collaborators.includes(collaborator._id)) {
            return res.status(400).json({ message: 'User is already a collaborator.' });
        }

        project.collaborators.push(collaborator._id);
        await project.save();

        res.status(200).json({ message: 'Collaborator added successfully!', project });
    } catch (error) {
        console.log(error);
        res.status(500).json({ message: 'Error adding collaborator', error: error.message });
    }
};

const uploadTestFile = async (req, res) => {
    const { projectId } = req.params;
    const ownerEmail = req.email;

    try {
        const project = await Project.findById(projectId);
        if (!project) {
            return res.status(404).json({ message: 'Project not found.' });
        }

        const user = await User.findOne({ email: ownerEmail });
        if (!user || !project.owner.equals(user._id)) {
            return res.status(403).json({ message: 'Only the project owner can upload test files.' });
        }

        if (!req.files || !req.files.testFile) {
            return res.status(400).json({ message: 'No test file uploaded.' });
        }

        const testFile = req.files.testFile;
        const userFolderPath = path.join(__dirname, '..', 'data', 'users', user.username);
        const projectFolderPath = path.join(userFolderPath, project.name);
        const testFilePath = path.join(projectFolderPath, 'test');

        if (!fs.existsSync(userFolderPath)) {
            fs.mkdirSync(userFolderPath, { recursive: true });
        }

        if (!fs.existsSync(projectFolderPath)) {
            fs.mkdirSync(projectFolderPath, { recursive: true });
        }

        testFile.mv(testFilePath, (err) => {
            if (err) {
                return res.status(500).json({ message: 'Error uploading test file', error: err });
            }

            res.status(200).json({ message: 'Test file uploaded successfully!' });
        });
    } catch (error) {
        res.status(400).json({ message: 'Error uploading test file', error });
    }
};
const getModel = async (req, res) => {
    const { projectName } = req.params;
    const userEmail = req.email;

    try {
        const project = await Project.findOne({ name: projectName }).populate('owner').populate('collaborators');
        if (!project) {
            return res.status(404).json({ message: 'Project not found.' });
        }

        const user = await User.findOne({ email: userEmail });
        if (!user) {
            return res.status(404).json({ message: 'User not found.' });
        }

        const isOwner = project.owner.equals(user._id);
        const isCollaborator = project.collaborators.some(collaborator => collaborator.equals(user._id));
        const isPublic = !project.isPrivate;

        if (!isOwner && !isCollaborator && !isPublic) {
            return res.status(403).json({ message: 'Access denied.' });
        }

        const userFolderPath = path.join(__dirname, '..', 'py', 'users', project.owner.username);
        const projectFolderPath = path.join(userFolderPath, projectName);
        const modelConfigPath = path.join(projectFolderPath, 'model_config.json');
        const modelWeightsPath = path.join(projectFolderPath, 'model.weights.h5');

        // Add debugging logs
        console.log(`Looking for model files in: ${projectFolderPath}`);
        console.log(`Model config path: ${modelConfigPath}`);
        console.log(`Model weights path: ${modelWeightsPath}`);
        console.log(`Config exists: ${fs.existsSync(modelConfigPath)}`);
        console.log(`Weights exists: ${fs.existsSync(modelWeightsPath)}`);
        
        // List all files in the project directory for debugging
        if (fs.existsSync(projectFolderPath)) {
            const filesInProject = fs.readdirSync(projectFolderPath);
            console.log(`Files in project directory: ${filesInProject.join(', ')}`);
        } else {
            console.log(`Project directory does not exist: ${projectFolderPath}`);
        }

        if (!fs.existsSync(modelConfigPath) || !fs.existsSync(modelWeightsPath)) {
            return res.status(404).json({ message: 'Model files not found.' });
        }

        const zip = new AdmZip();
        zip.addLocalFile(modelConfigPath);
        zip.addLocalFile(modelWeightsPath);

        const zipPath = path.join(projectFolderPath, 'model.zip');
        zip.writeZip(zipPath);

        res.download(zipPath, 'model.zip', (err) => {
            if (err) {
                console.error(`Error downloading zip file: ${err.message}`);
                return res.status(500).json({ message: 'Error downloading zip file', error: err.message });
            }

            // Remove the temporary zip file after download
            fs.unlinkSync(zipPath);
        });
    } catch (error) {
        console.error('Error in getModel:', error);
        res.status(400).json({ message: 'Error retrieving model', error: error.message });
    }
};


const testModel = async (req, res) => {
    const { projectName } = req.params;
    const ownerEmail = req.email;

    try {
        const project = await Project.findOne({ name: projectName });
        if (!project) {
            return res.status(404).json({ message: 'Project not found.' });
        }

        const user = await User.findOne({ email: ownerEmail });
        if (!user || !project.owner.equals(user._id)) {
            return res.status(403).json({ message: 'Only the project owner can test the model.' });
        }
        const username = user.username;

        const scriptPath = path.join(__dirname, '..', 'py', 'test.py');
        const venvActivatePath = path.join(__dirname, '..', 'py', '.venv', 'bin', 'activate');
        const command = `cd ${path.join(__dirname, '..', 'py')} && . ${venvActivatePath} && python ${scriptPath} ${username} ${projectName}`;

        exec(command, async (error, stdout, stderr) => {
            if (error) {
                console.error(`Error executing Python script: ${error.message}`);
                return res.status(500).json({ message: 'Error executing Python script', error: error.message });
            }

            console.log(`Python script output: ${stdout}`);
            console.error(`Python script error output: ${stderr}`);

            const accuracyFilePath = path.join(__dirname, '..', 'py', 'users', username, projectName, 'accuracy.txt');
            if (!fs.existsSync(accuracyFilePath)) {
                return res.status(404).json({ message: 'Accuracy file not found.' });
            }
            console.log("Accuracy : ");
            const accuracy = fs.readFileSync(accuracyFilePath, 'utf8').trim();
            project.accuracy = parseFloat(accuracy);

            try {
                await project.save();
                res.status(200).json({ message: 'Model tested successfully!', accuracy: project.accuracy });
            } catch (error) {
                res.status(400).json({ message: 'Error updating project accuracy', error });
            }
        });
    } catch (error) {
        res.status(400).json({ message: 'Error testing model', error });
    }
};


const contribute = async (req, res) => {
    const { projectName } = req.params;
    const contributorEmail = req.email;

    try {
        // Find project and populate owner details
        const project = await Project.findOne({ name: projectName }).populate('owner');
        if (!project) {
            return res.status(404).json({ message: 'Project not found.' });
        }

        // Find contributor
        const contributor = await User.findOne({ email: contributorEmail });
        if (!contributor) {
            return res.status(404).json({ message: 'Contributor not found.' });
        }

        // Check if user is owner or collaborator
        const isOwner = project.owner._id.equals(contributor._id);
        const isCollaborator = project.collaborators.some(collab => collab.equals(contributor._id));
        
        if (!isOwner && !isCollaborator) {
            return res.status(403).json({ message: 'Not authorized to contribute to this project.' });
        }

        // Verify file upload
        if (!req.files || !req.files.file) {
            return res.status(400).json({ message: 'No model file uploaded.' });
        }

        const modelFile = req.files.file;
        const sha1Hash = req.body.sha1;

        if (!sha1Hash) {
            return res.status(400).json({ message: 'SHA1 hash is required.' });
        }

        // Setup directories
        const contribPath = path.join(
            __dirname, 
            '..', 
            'py',
            'users',
            project.owner.username,
            project.name,
            'contrib'
        );

        // Create directories if they don't exist
        fs.mkdirSync(contribPath, { recursive: true });

        // Save the model file
        const modelFilePath = path.join(contribPath, `${sha1Hash}.weights.h5`);
        await modelFile.mv(modelFilePath);

        // Run contribution script
        const projectDir = path.join(__dirname, '..', 'py');
        const venvActivate = path.join(projectDir, '.venv', 'bin', 'activate');
        const scriptPath = path.join(projectDir, 'contribution.py');
        const username = project.owner.username;

        const command = `cd ${projectDir} && . ${venvActivate} && python ${scriptPath} ${username} ${projectName} ${sha1Hash}`;

        exec(command, async (error, stdout, stderr) => {
            if (error) {
                console.error(`Error executing Python script: ${error.message}`);
                return res.status(500).json({ 
                    message: 'Error executing Python script', 
                    error: error.message,
                    stdout: stdout,
                    stderr: stderr
                });
            }

            console.log(`Python script output: ${stdout}`);
            console.error(`Python script error output: ${stderr}`);

            try {
                // Create new contribution record
                const newContribution = new Contribution({
                    project: project._id,
                    user: contributor._id,
                    hash: sha1Hash
                });

                await newContribution.save();

                // Update project's contributions array
                await Project.findByIdAndUpdate(
                    project._id,
                    { $push: { contributions: newContribution._id } }
                );

                res.status(200).json({ 
                    message: 'Contribution processed successfully',
                    contributionId: newContribution._id
                });
            } catch (dbError) {
                console.error('Database error:', dbError);
                res.status(500).json({ 
                    message: 'Error saving contribution record', 
                    error: dbError.message 
                });
            }
        });

    } catch (error) {
        console.error('Server error:', error);
        res.status(500).json({ 
            message: 'Server error', 
            error: error.message 
        });
    }
};


const getJson = async (req, res) => {
    const { projectName } = req.params;
    const userEmail = req.email; // Ensure that `req.email` is set by your authentication middleware

    try {
        // Fetch the project from the database
        const project = await Project.findOne({ name: projectName })
            .populate('owner')
            .populate('collaborators');

        if (!project) {
            return res.status(404).json({ message: 'Project not found.' });
        }

        // Fetch the user from the database
        const user = await User.findOne({ email: userEmail });
        if (!user) {
            return res.status(404).json({ message: 'User not found.' });
        }

        // Access Control: Check if the user is the owner, a collaborator, or if the project is public
        const isOwner = project.owner.equals(user._id);
        const isCollaborator = project.collaborators.some(collaborator => collaborator.equals(user._id));
        const isPublic = !project.isPrivate;

        if (!isOwner && !isCollaborator && !isPublic) {
            return res.status(403).json({ message: 'Access denied.' });
        }

        // Define paths
        const userFolderPath = path.join(__dirname, '..', 'py', 'users', project.owner.username);
        const projectFolderPath = path.join(userFolderPath, projectName);
        const modelConfigPath = path.join(projectFolderPath, 'model_config.json');

        // Check if model_config.json exists
        if (!fs.existsSync(modelConfigPath)) {
            return res.status(404).json({ message: 'model_config.json not found.' });
        }

        // Send the model_config.json file
        return res.sendFile(modelConfigPath, (err) => {
            if (err) {
                console.error('Error sending model_config.json:', err.message);
                return res.status(500).json({ message: 'Error sending the model configuration.', error: err.message });
            }
        });

    } catch (error) {
        console.error('Error in getJson:', error);
        return res.status(500).json({ message: 'Server error.', error: error.message });
    }
};

module.exports = {
    createProject,
    deleteProject,
    getAllProjects,
    updateProject,
    addCollaborator,
    uploadTestFile,
    getModel,
    testModel,
    contribute,
    getJson
};