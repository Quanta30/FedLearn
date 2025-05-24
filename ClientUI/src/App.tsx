import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import ProjectList from './components/ProjectList';
import AuthComponent from './components/AuthScreen';
import Contrib from './components/Contrib';
import { Project } from './types';

function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem("token"));
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [contribScreen, setContribScreen] = useState<boolean>(false);

  // Reset contribution screen when project changes
  React.useEffect(() => {
    setContribScreen(false);
  }, [currentProject]);

  return (
    <>
      {token ? (
        <>
          <Navbar setToken={setToken} />
          {currentProject ? (
            contribScreen ? (
              <Contrib
                token={token}
                currentProject={currentProject}
                setContribScreen={setContribScreen}
              />
            ) : (
              <Dashboard
                token={token}
                currentProject={currentProject}
                setCurrentProject={setCurrentProject}
                setContribScreen={setContribScreen}
              />
            )
          ) : (
            <ProjectList token={token} setCurrentProject={setCurrentProject} />
          )}
        </>
      ) : (
        <AuthComponent setToken={setToken} />
      )}
    </>
  );
}

export default App;