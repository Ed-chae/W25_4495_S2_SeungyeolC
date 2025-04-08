// src/components/FileUpload.js
import React, { useState } from "react";
import axios from "../services/api";
import { motion } from "framer-motion";

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage("");
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("⚠️ Please select an Excel file before uploading.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);

    try {
      const response = await axios.post("/upload/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setMessage("✅ " + response.data.message);
    } catch (error) {
      setMessage(
        "❌ Upload failed: " +
          (error.response?.data?.detail || error.message)
      );
    } finally {
      setUploading(false);
    }
  };

  return (
    <motion.div
      className="card max-w-2xl mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-bold text-indigo-700 mb-3">📂 Upload Excel File</h2>
      <p className="text-sm text-gray-600 mb-4">
        Select and upload your Excel file (.xlsx or .xls) to begin analysis.
      </p>

      <input
        type="file"
        accept=".xlsx,.xls"
        onChange={handleFileChange}
        className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0
                   file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700
                   hover:file:bg-indigo-100 mb-4"
      />

      <button
        onClick={handleUpload}
        disabled={uploading}
        className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition disabled:opacity-50"
      >
        {uploading ? "Uploading..." : "Upload"}
      </button>

      {message && (
        <p className={`mt-3 text-sm ${message.startsWith("✅") ? "text-green-600" : "text-red-500"}`}>
          {message}
        </p>
      )}
    </motion.div>
  );
};

export default FileUpload;
