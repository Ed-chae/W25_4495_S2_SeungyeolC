import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";  // ✅ FastAPI Backend
const FLASK_BASE_URL = "http://127.0.0.1:5000";  // ✅ Flask Forecasting API

// ✅ Fetch Analytics Data
export const fetchAnalytics = async () => {
  try {
    console.log("Fetching analytics data...");
    const response = await axios.get(`${API_BASE_URL}/analytics/`);
    console.log("Analytics Data:", response.data);
    return response.data;
  } catch (error) {
    console.error("Error fetching analytics:", error.response?.data || error.message);
    return null;
  }
};

// ✅ Upload an Excel File and Trigger Forecast Update
export const uploadFile = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    console.log("Uploading file...", file);

    const response = await axios.post(`${FLASK_BASE_URL}/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    console.log("Upload Success:", response.data);
    
    return await fetchForecast();
  } catch (error) {
    console.error("Error uploading file:", error);
    return { error: "File upload failed." };
  }
};

// ✅ Fetch Revenue Forecast Data
export const fetchForecast = async () => {
  try {
    console.log("Fetching revenue forecast...");
    const response = await axios.get(`${FLASK_BASE_URL}/forecast`);
    console.log("Forecast Data:", response.data);
    return response.data;
  } catch (error) {
    console.error("Error fetching forecast data:", error);
    return [];
  }
};
