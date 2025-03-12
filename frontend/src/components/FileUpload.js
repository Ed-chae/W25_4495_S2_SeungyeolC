// Filename: FileUpload.js

import React, { useState } from "react";
import { uploadFile } from "../services/api";

const FileUpload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  // ✅ Handle file selection
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setMessage("");
  };

  // ✅ Handle file upload
  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a file to upload.");
      return;
    }

    setMessage("Uploading...");

    const response = await uploadFile(file);

    if (response.error) {
      setMessage("❌ Upload failed.");
    } else {
      setMessage("✅ Upload successful!");
      onUploadSuccess(); // Refresh dashboard after upload
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px", border: "1px solid #ddd", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
      <h3>📁 Upload Excel File</h3>
      <input type="file" accept=".xlsx" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: "10px", padding: "5px 10px", cursor: "pointer" }}>Upload</button>
      <p>{message}</p>
    </div>
  );
};

export default FileUpload;
