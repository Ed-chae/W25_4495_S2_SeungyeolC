// src/components/ResetButton.js
import React, { useState } from "react";
import { resetSalesData, resetRestaurantOrders } from "../services/resetApi";
import { motion } from "framer-motion";

const ResetButton = () => {
  const [message, setMessage] = useState("");

  const handleReset = async () => {
    try {
      await resetSalesData();
      await resetRestaurantOrders();
      setMessage("✅ All data has been reset successfully.");
    } catch (err) {
      console.error("Error resetting data:", err);
      setMessage("❌ Failed to reset data.");
    }
  };

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h2 className="text-lg font-semibold text-red-600 mb-2">🧹 Reset Data</h2>
      <p className="text-sm text-gray-600 mb-3">
        This will permanently delete all uploaded sales and restaurant data.
      </p>
      <button
        onClick={handleReset}
        className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition"
      >
        🗑️ Reset All Data
      </button>
      {message && <p className="mt-3 text-sm">{message}</p>}
    </motion.div>
  );
};

export default ResetButton;
