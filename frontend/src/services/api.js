// Filename: api.js

import axios from "axios";

// ✅ Backend API Base URL (Ensure this is correct for your environment)
const BASE_URL = "http://localhost:8000";  // Change if backend is running elsewhere

// ✅ Fetch analytics data from backend
export const fetchAnalytics = async () => {
  try {
    console.log("Fetching analytics data...");
    const response = await axios.get(`${BASE_URL}/analytics/`);
    console.log("Analytics Data:", response.data);
    return response.data;
  } catch (error) {
    console.error("Error fetching analytics:", error.response?.data || error.message);
    return null;
  }
};

// ✅ Upload an Excel file to the backend
export const uploadFile = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    console.log("Uploading file...", file);

    const response = await axios.post(`${BASE_URL}/upload/`, formData, {
      headers: { 
        "Content-Type": "multipart/form-data",
      },
    });

    console.log("Upload Success:", response.data);
    return response.data;
  } catch (error) {
    console.error("Error uploading file:", error.response?.data || error.message);
    return { error: "File upload failed." };
  }
};
