import React, { useState } from "react";
import axios from "../services/api";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

const ResetButton = () => {
  const [loading, setLoading] = useState(false);
  const [resetStatus, setResetStatus] = useState("");

  const handleReset = async () => {
    setLoading(true);
    setResetStatus("");

    try {
      const salesRes = await axios.delete("/reset-sales-data/");
      const ordersRes = await axios.delete("/reset-restaurant-orders/");

      setResetStatus("✅ Data reset successfully.");
    } catch (error) {
      console.error("Reset failed:", error);
      setResetStatus("❌ Reset failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="mt-4"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Button
        onClick={handleReset}
        disabled={loading}
        className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded shadow"
      >
        {loading ? "Resetting..." : "Reset Data"}
      </Button>

      {resetStatus && (
        <p className={`mt-2 text-sm ${resetStatus.includes("✅") ? "text-green-600" : "text-red-500"}`}>
          {resetStatus}
        </p>
      )}
    </motion.div>
  );
};

export default ResetButton;
