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
        // Log incoming request data for debugging
        console.log('Received createProject request:');
        console.log('Request body:', req.body);
        console.log('Owner email:', owner);

        // Validate required fields
        if (!name || !name.trim()) {
            return res.status(400).json({ message: 'Project name is required.' });
        }

        if (!owner) {
            return res.status(400).json({ message: 'User authentication required.' });
        }

        const existingProject = await Project.findOne({ name });
        if (existingProject) {
            return res.status(400).json({ message: 'Project name must be unique.' });
        }

        const user = await User.findOne({ email: owner });
        if (!user) {
            return res.status(404).json({ message: 'User not found.' });
        }

        // Set default values for MNIST training if parameters are undefined
        // Ensure proper type conversion for numeric values
        const projectConfig = {
            activation_function: activation_function || 'relu',
            dropout_rate: (dropout_rate !== undefined && dropout_rate !== null && dropout_rate !== '') ? 
                parseFloat(dropout_rate) : 0.2,
            combining_method: combining_method || 'average',
            input_shape: input_shape || '28,28',
            num_layers: (num_layers !== undefined && num_layers !== null && num_layers !== '') ? 
                parseInt(num_layers) : 3,
            units_per_layer: (units_per_layer !== undefined && units_per_layer !== null && units_per_layer !== '') ? 
                parseInt(units_per_layer) : 128,
            num_classes: (num_classes !== undefined && num_classes !== null && num_classes !== '') ? 
                parseInt(num_classes) : 10
        };

        console.log('Processed config:', projectConfig);

        // Validate numeric values
        if (isNaN(projectConfig.dropout_rate) || projectConfig.dropout_rate < 0 || projectConfig.dropout_rate > 1) {
            return res.status(400).json({ message: 'Dropout rate must be a number between 0 and 1.' });
        }

        if (isNaN(projectConfig.num_layers) || projectConfig.num_layers < 1) {
            return res.status(400).json({ message: 'Number of layers must be a positive integer.' });
        }

        if (isNaN(projectConfig.units_per_layer) || projectConfig.units_per_layer < 1) {
            return res.status(400).json({ message: 'Units per layer must be a positive integer.' });
        }

        if (isNaN(projectConfig.num_classes) || projectConfig.num_classes < 1) {
            return res.status(400).json({ message: 'Number of classes must be a positive integer.' });
        }

        const newProject = new Project({
            name: name.trim(),
            description: description || '',
            owner: user._id,
            isPrivate: isPrivate !== undefined ? Boolean(isPrivate) : false,
            activation_function: projectConfig.activation_function,
            dropout_rate: projectConfig.dropout_rate,
            combining_method: projectConfig.combining_method,
            input_shape: projectConfig.input_shape,
            num_layers: projectConfig.num_layers,
            units_per_layer: projectConfig.units_per_layer,
            num_classes: projectConfig.num_classes
        });

        console.log('Created project object:', newProject);

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

        // Create config.txt file with validated values
        const configContent = `
activation_function=${projectConfig.activation_function}
dropout_rate=${projectConfig.dropout_rate}
combining_method=${projectConfig.combining_method}
input_shape=${projectConfig.input_shape}
num_layers=${projectConfig.num_layers}
units_per_layer=${projectConfig.units_per_layer}
num_classes=${projectConfig.num_classes}
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
        console.error('Error in createProject:', error);
        res.status(400).json({ 
            message: 'Error creating project', 
            error: error.message,
            details: error.stack 
        });
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
    const { projectName } = req.params;
    const ownerEmail = req.email;

    try {
        const project = await Project.findOne({ name: projectName });
        if (!project) {
            return res.status(404).json({ message: 'Project not found.' });
        }

        const user = await User.findOne({ email: ownerEmail });
        if (!user || !project.owner.equals(user._id)) {
            return res.status(403).json({ message: 'Only the project owner can upload test files.' });
        }

        if (!req.files || !req.files.testSetZip) {
            return res.status(400).json({ message: 'No test file uploaded. Please upload a zip file containing your test dataset.' });
        }

        const testZipFile = req.files.testSetZip;
        
        // Debug: Log file information
        console.log('Uploaded test file details:');
        console.log('- Name:', testZipFile.name);
        console.log('- Size:', testZipFile.size);
        console.log('- MIME type:', testZipFile.mimetype);

        // Get file buffer similar to contribute function
        let fileBuffer;
        if (testZipFile.data && testZipFile.data.length > 0) {
            fileBuffer = testZipFile.data;
        } else if (testZipFile.tempFilePath && fs.existsSync(testZipFile.tempFilePath)) {
            fileBuffer = fs.readFileSync(testZipFile.tempFilePath);
            console.log('Read test file from temp path:', testZipFile.tempFilePath);
        } else {
            return res.status(400).json({ message: 'Uploaded test file data is not accessible. Please try uploading again.' });
        }

        // Validate zip file
        const isZipFile = (data) => {
            if (!data || data.length < 4) return false;
            return (data[0] === 0x50 && data[1] === 0x4B && 
                   (data[2] === 0x03 && data[3] === 0x04 ||
                    data[2] === 0x05 && data[3] === 0x06 ||
                    data[2] === 0x07 && data[3] === 0x08));
        };

        if (!isZipFile(fileBuffer)) {
            return res.status(400).json({ message: 'Invalid file format. Please upload a zip file containing your test dataset.' });
        }

        const userFolderPath = path.join(__dirname, '..', 'data', 'users', user.username);
        const projectFolderPath = path.join(userFolderPath, project.name);
        const testDirPath = path.join(projectFolderPath, 'test');

        // Create directories if they don't exist
        fs.mkdirSync(userFolderPath, { recursive: true });
        fs.mkdirSync(projectFolderPath, { recursive: true });
        
        // Remove existing test directory if it exists
        if (fs.existsSync(testDirPath)) {
            fs.rmSync(testDirPath, { recursive: true, force: true });
        }
        fs.mkdirSync(testDirPath, { recursive: true });

        try {
            // Save zip file temporarily
            const tempZipPath = path.join(projectFolderPath, 'temp_test.zip');
            fs.writeFileSync(tempZipPath, fileBuffer);

            console.log(`Saved temp test zip: ${tempZipPath}`);
            console.log(`Temp file size: ${fs.statSync(tempZipPath).size}`);

            // Extract the zip file
            const zip = new AdmZip(tempZipPath);
            const zipEntries = zip.getEntries();
            
            console.log('Test zip entries found:');
            zipEntries.forEach(entry => {
                console.log(`- ${entry.entryName} (${entry.header.size} bytes)`);
            });

            // Extract to test directory
            zip.extractAllTo(testDirPath, true);

            // Clean up temp file
            fs.unlinkSync(tempZipPath);

            // Verify extraction - check if we have labeled folders with images
            const extractedItems = fs.readdirSync(testDirPath);
            console.log('Extracted items in test directory:', extractedItems);

            // Check if we have the expected structure (labeled folders)
            let hasValidStructure = false;
            for (const item of extractedItems) {
                const itemPath = path.join(testDirPath, item);
                if (fs.statSync(itemPath).isDirectory()) {
                    const filesInFolder = fs.readdirSync(itemPath);
                    if (filesInFolder.length > 0) {
                        hasValidStructure = true;
                        console.log(`Found label folder '${item}' with ${filesInFolder.length} files`);
                    }
                }
            }

            if (!hasValidStructure) {
                fs.rmSync(testDirPath, { recursive: true, force: true });
                return res.status(400).json({ 
                    message: 'Invalid test dataset structure. Please ensure your zip contains labeled folders with images inside each folder.' 
                });
            }

            console.log(`Successfully extracted test dataset to: ${testDirPath}`);
            res.status(200).json({ message: 'Test files uploaded successfully!' });

        } catch (extractError) {
            console.error('Error extracting test file:', extractError);
            
            // Clean up on error
            if (fs.existsSync(testDirPath)) {
                fs.rmSync(testDirPath, { recursive: true, force: true });
            }
            
            return res.status(400).json({ 
                message: 'Error extracting test file: ' + extractError.message 
            });
        }

    } catch (error) {
        console.error('Error in uploadTestFile:', error);
        res.status(400).json({ message: 'Error uploading test file', error: error.message });
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
        const modelDir = path.join(projectFolderPath, 'model');

        // Add debugging logs
        console.log(`Looking for model files in: ${projectFolderPath}`);
        
        // Check if project directory exists
        if (!fs.existsSync(projectFolderPath)) {
            console.log(`Project directory does not exist: ${projectFolderPath}`);
            return res.status(404).json({ message: 'Project directory not found.' });
        }

        // List all files in the project directory for debugging
        const filesInProject = fs.readdirSync(projectFolderPath);
        console.log(`Files in project directory: ${filesInProject.join(', ')}`);

        // Check if model directory exists
        if (!fs.existsSync(modelDir)) {
            console.log(`Model directory does not exist: ${modelDir}`);
            return res.status(404).json({ message: 'No model has been created yet. Make a contribution first.' });
        }

        const filesInModelDir = fs.readdirSync(modelDir);
        console.log(`Files in model directory: ${filesInModelDir.join(', ')}`);

        // Check for required TensorFlow.js files
        const modelJsonPath = path.join(modelDir, 'model.json');
        if (!fs.existsSync(modelJsonPath)) {
            return res.status(404).json({ message: 'Model files not found.' });
        }

        // Create zip of the entire model directory
        const zip = new AdmZip();
        zip.addLocalFolder(modelDir, 'model');

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

        // Check if model exists
        const userFolderPath = path.join(__dirname, '..', 'py', 'users', user.username);
        const projectFolderPath = path.join(userFolderPath, projectName);
        const modelDir = path.join(projectFolderPath, 'model');

        console.log(`Checking model directory: ${modelDir}`);
        if (!fs.existsSync(modelDir)) {
            return res.status(404).json({ message: 'No model found. Please make a contribution first.' });
        }

        // Verify model files exist
        const modelJsonPath = path.join(modelDir, 'model.json');
        if (!fs.existsSync(modelJsonPath)) {
            return res.status(404).json({ message: 'Model files are incomplete. Please make a new contribution.' });
        }

        // Check if test data exists
        const testDataPath = path.join(__dirname, '..', 'data', 'users', user.username, projectName, 'test');
        console.log(`Checking test data directory: ${testDataPath}`);
        
        if (!fs.existsSync(testDataPath)) {
            return res.status(404).json({ message: 'No test data found. Please upload test files first.' });
        }

        // Verify test data structure and log details
        const testItems = fs.readdirSync(testDataPath);
        console.log(`Test data items found: ${testItems.join(', ')}`);
        
        let hasValidTestStructure = false;
        let labelFolders = [];
        
        for (const item of testItems) {
            const itemPath = path.join(testDataPath, item);
            if (fs.statSync(itemPath).isDirectory()) {
                const filesInFolder = fs.readdirSync(itemPath);
                if (filesInFolder.length > 0) {
                    hasValidTestStructure = true;
                    labelFolders.push(`${item} (${filesInFolder.length} files)`);
                }
            }
        }

        console.log(`Label folders found: ${labelFolders.join(', ')}`);

        if (!hasValidTestStructure) {
            return res.status(400).json({ message: 'Invalid test data structure. Please upload labeled folders with images.' });
        }

        const username = user.username;
        const scriptPath = path.join(__dirname, '..', 'py', 'test.py');
        const venvActivatePath = path.join(__dirname, '..', 'py', '.venv', 'bin', 'activate');
        
        // Add environment variable to suppress TensorFlow warnings
        const command = `cd ${path.join(__dirname, '..', 'py')} && . ${venvActivatePath} && TF_CPP_MIN_LOG_LEVEL=2 python ${scriptPath} ${username} ${projectName}`;

        console.log(`Executing test command: ${command}`);

        exec(command, { timeout: 300000 }, async (error, stdout, stderr) => {
            console.log(`Python script stdout: ${stdout}`);
            console.log(`Python script stderr: ${stderr}`);
            
            if (error) {
                console.error(`Error executing Python script: ${error.message}`);
                
                // Check if it's a timeout error
                if (error.code === 'ETIMEDOUT') {
                    return res.status(408).json({ 
                        message: 'Test script timed out. The model or test data might be too large.',
                        error: 'Timeout after 5 minutes'
                    });
                }
                
                return res.status(500).json({ 
                    message: 'Error executing Python script', 
                    error: error.message,
                    stdout: stdout,
                    stderr: stderr 
                });
            }

            // Check if the script completed successfully by looking for accuracy file
            const accuracyFilePath = path.join(__dirname, '..', 'py', 'users', username, projectName, 'accuracy.txt');
            console.log(`Looking for accuracy file at: ${accuracyFilePath}`);
            
            if (!fs.existsSync(accuracyFilePath)) {
                console.log(`Accuracy file not found. Script may have failed.`);
                
                // Check if there are any error indicators in stdout/stderr
                const errorIndicators = ['error', 'exception', 'failed', 'traceback'];
                const hasError = errorIndicators.some(indicator => 
                    stdout.toLowerCase().includes(indicator) || stderr.toLowerCase().includes(indicator)
                );
                
                if (hasError) {
                    return res.status(500).json({ 
                        message: 'Test script failed with errors.',
                        stdout: stdout,
                        stderr: stderr
                    });
                }
                
                return res.status(404).json({ 
                    message: 'Test completed but accuracy file was not generated. Check the test script logs.',
                    stdout: stdout,
                    stderr: stderr
                });
            }

            try {
                console.log("Reading accuracy from:", accuracyFilePath);
                const accuracy = fs.readFileSync(accuracyFilePath, 'utf8').trim();
                console.log(`Raw accuracy value: "${accuracy}"`);
                
                const accuracyValue = parseFloat(accuracy);

                if (isNaN(accuracyValue)) {
                    return res.status(400).json({ 
                        message: `Invalid accuracy value: "${accuracy}". Expected a number.`,
                        stdout: stdout,
                        stderr: stderr
                    });
                }

                project.accuracy = accuracyValue;

                await project.save();
                console.log(`Successfully saved accuracy: ${accuracyValue}`);
                
                res.status(200).json({ 
                    message: 'Model tested successfully!', 
                    accuracy: project.accuracy,
                    details: {
                        testDataPath: testDataPath,
                        labelFolders: labelFolders,
                        stdout: stdout
                    }
                });
                
            } catch (saveError) {
                console.error('Error saving accuracy:', saveError);
                res.status(400).json({ 
                    message: 'Error updating project accuracy', 
                    error: saveError.message,
                    accuracy: accuracy
                });
            }
        });
    } catch (error) {
        console.error('Error in testModel:', error);
        res.status(400).json({ message: 'Error testing model', error: error.message });
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

        // Verify file upload - expecting a zip file containing complete TensorFlow.js model
        if (!req.files || !req.files.file) {
            return res.status(400).json({ message: 'No model file uploaded. Please upload a zip file containing model.json and weights.bin files.' });
        }

        const modelFile = req.files.file;
        const sha1Hash = req.body.sha1;

        if (!sha1Hash) {
            return res.status(400).json({ message: 'SHA1 hash is required.' });
        }

        // Debug: Log file information
        console.log('Uploaded file details:');
        console.log('- Name:', modelFile.name);
        console.log('- Size:', modelFile.size);
        console.log('- MIME type:', modelFile.mimetype);
        console.log('- Data type:', typeof modelFile.data);
        console.log('- Data length:', modelFile.data ? modelFile.data.length : 'undefined');
        console.log('- Has mv function:', typeof modelFile.mv);

        // Check if file has data - use different methods to access file data
        let fileBuffer;
        if (modelFile.data && modelFile.data.length > 0) {
            fileBuffer = modelFile.data;
        } else if (modelFile.tempFilePath && fs.existsSync(modelFile.tempFilePath)) {
            // File might be stored as a temp file
            fileBuffer = fs.readFileSync(modelFile.tempFilePath);
            console.log('Read file from temp path:', modelFile.tempFilePath);
        } else {
            return res.status(400).json({ message: 'Uploaded file data is not accessible. Please try uploading again.' });
        }

        console.log('File buffer length:', fileBuffer.length);

        // More flexible file validation - check for zip magic number
        const isZipFile = (data) => {
            if (!data || data.length < 4) return false;
            // Check for ZIP file magic numbers
            return (data[0] === 0x50 && data[1] === 0x4B && 
                   (data[2] === 0x03 && data[3] === 0x04 ||  // Local file header
                    data[2] === 0x05 && data[3] === 0x06 ||  // End of central directory
                    data[2] === 0x07 && data[3] === 0x08));  // Spanned archive
        };

        if (!isZipFile(fileBuffer)) {
            // If not a zip file, check if it's JSON (maybe they uploaded just model.json)
            try {
                const fileContent = fileBuffer.toString('utf8');
                JSON.parse(fileContent);
                return res.status(400).json({ 
                    message: 'You uploaded a JSON file. Please upload a complete zip file containing both model.json and weights.bin files.' 
                });
            } catch {
                return res.status(400).json({ 
                    message: 'Invalid file format. Please upload a zip file containing the complete TensorFlow.js model (model.json and weights.bin).' 
                });
            }
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

        // Create a directory for this contribution
        const contributionDir = path.join(contribPath, `${sha1Hash}.tfjs`);
        fs.mkdirSync(contributionDir, { recursive: true });

        try {
            // Save the uploaded file temporarily to debug
            const tempPath = path.join(contribPath, `temp_${sha1Hash}.zip`);
            fs.writeFileSync(tempPath, fileBuffer);
            
            console.log(`Saved temp file: ${tempPath}`);
            console.log(`Temp file size: ${fs.statSync(tempPath).size}`);

            // Extract the zip file
            const zip = new AdmZip(tempPath);
            const zipEntries = zip.getEntries();
            
            console.log('Zip entries found:');
            zipEntries.forEach(entry => {
                console.log(`- ${entry.entryName} (${entry.header.size} bytes)`);
            });

            zip.extractAllTo(contributionDir, true);

            // Clean up temp file
            fs.unlinkSync(tempPath);

            // Validate that the extracted files include the required TensorFlow.js files
            const modelJsonPath = path.join(contributionDir, 'model.json');
            if (!fs.existsSync(modelJsonPath)) {
                // Cleanup and return error
                fs.rmSync(contributionDir, { recursive: true, force: true });
                return res.status(400).json({ 
                    message: 'Invalid model file. The zip must contain model.json file.' 
                });
            }

            // Check for weights files by reading model.json
            const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
            const weightsManifest = modelJson.weightsManifest || [];
            const missingWeights = [];

            for (const manifest of weightsManifest) {
                for (const weightPath of manifest.paths || []) {
                    const weightFile = path.join(contributionDir, weightPath);
                    if (!fs.existsSync(weightFile)) {
                        missingWeights.push(weightPath);
                    }
                }
            }

            if (missingWeights.length > 0) {
                // Cleanup and return error
                fs.rmSync(contributionDir, { recursive: true, force: true });
                return res.status(400).json({ 
                    message: `Invalid model file. Missing weight files: ${missingWeights.join(', ')}` 
                });
            }

            console.log(`Successfully extracted TensorFlow.js model to: ${contributionDir}`);
            
        } catch (extractError) {
            console.error('Error extracting model file:', extractError);
            
            // More specific error handling
            if (extractError.message.includes('Invalid or unsupported zip format')) {
                return res.status(400).json({ 
                    message: 'The uploaded file is not a valid zip file. Please ensure you are uploading a properly compressed zip file containing your TensorFlow.js model.' 
                });
            }
            
            // Cleanup on error
            if (fs.existsSync(contributionDir)) {
                fs.rmSync(contributionDir, { recursive: true, force: true });
            }
            return res.status(400).json({ 
                message: 'Error extracting model file: ' + extractError.message 
            });
        }

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

        // Define paths for TensorFlow.js format
        const userFolderPath = path.join(__dirname, '..', 'py', 'users', project.owner.username);
        const projectFolderPath = path.join(userFolderPath, projectName);
        const modelJsonPath = path.join(projectFolderPath, 'model', 'model.json');

        // Check if model.json exists
        if (!fs.existsSync(modelJsonPath)) {
            return res.status(404).json({ message: 'model.json not found.' });
        }

        // Send the model.json file
        return res.sendFile(modelJsonPath, (err) => {
            if (err) {
                console.error('Error sending model.json:', err.message);
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