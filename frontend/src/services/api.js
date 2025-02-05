// Filename: api.js

import axios from "axios";

// ✅ Backend API Base URL (Modify if needed)
const BASE_URL = "http://192.168.1.76:8000";

// ✅ Fetch analytics data from backend
export const fetchAnalytics = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/analytics/`);
    return response.data;
  } catch (error) {
    console.error("Error fetching analytics:", error);
    return null;
  }
};

// ✅ Upload an Excel file to the backend
export const uploadFile = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(`${BASE_URL}/upload/`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    return response.data;
  } catch (error) {
    console.error("Error uploading file:", error);
    return { error: "File upload failed." };
  }
};
