// Filename: App.js

import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import Dashboard from "./components/Dashboard";

const App = () => {
  const [refresh, setRefresh] = useState(false);

  // ✅ Refresh dashboard after file upload
  const handleUploadSuccess = () => {
    setRefresh(!refresh);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ textAlign: "center" }}>📊 Intelligent Business Analytics</h1>
      <FileUpload onUploadSuccess={handleUploadSuccess} />
      <Dashboard key={refresh} />
    </div>
  );
};

export default App;
