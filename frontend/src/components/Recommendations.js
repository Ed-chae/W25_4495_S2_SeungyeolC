// src/components/Recommendations.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { motion } from "framer-motion";

const Recommendations = ({ userId = 1 }) => {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!userId) {
      setError("⚠️ No user ID provided.");
      setLoading(false);
      return;
    }

    axios
      .get(`/product-recommendations/?user_id=${userId}`)
      .then((res) => {
        setRecommendations(res.data);
        setError("");
      })
      .catch((err) => {
        console.error("Error fetching recommendations:", err);
        setError("❌ Failed to load recommendations.");
      })
      .finally(() => setLoading(false));
  }, [userId]);

  if (loading) return <p className="text-gray-500">Loading personalized recommendations...</p>;
  if (error) return <p className="text-red-500">{error}</p>;
  if (!recommendations) return <p>No recommendations available.</p>;

  return (
    <motion.div
      className="dashboard-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-bold text-purple-700 mb-4">🎯 Product Recommendations</h2>

      {recommendations.svd_recommendations?.length > 0 && (
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-700 mb-2">🧠 Based on Collaborative Filtering</h3>
          <div className="space-y-1 text-sm text-gray-800">
            {recommendations.svd_recommendations.map((item, idx) => (
              <div key={idx} className="p-2 bg-purple-50 rounded">{item}</div>
            ))}
          </div>
        </div>
      )}

      {recommendations.nn_recommendations?.length > 0 && (
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-700 mb-2">🔗 Based on Neural Network</h3>
          <div className="space-y-1 text-sm text-gray-800">
            {recommendations.nn_recommendations.map((item, idx) => (
              <div key={idx} className="p-2 bg-indigo-50 rounded">{item}</div>
            ))}
          </div>
        </div>
      )}

      {recommendations.svd_recommendations?.length === 0 &&
        recommendations.nn_recommendations?.length === 0 && (
          <p className="text-gray-500">No recommendations available for this user.</p>
        )}
    </motion.div>
  );
};

export default Recommendations;
