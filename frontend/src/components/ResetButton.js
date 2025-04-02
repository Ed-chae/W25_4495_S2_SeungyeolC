import React, { useState } from "react";
import { resetSalesData, resetRestaurantOrders } from "../services/resetApi";

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
    <div className="mt-4">
      <button
        onClick={handleReset}
        className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition"
      >
        🗑️ Reset All Data
      </button>
      {message && <p className="mt-2 text-sm text-gray-700">{message}</p>}
    </div>
  );
};

export default ResetButton;
