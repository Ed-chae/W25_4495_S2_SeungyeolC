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
      className="space-y-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-semibold text-purple-700">🎯 Product Recommendations</h2>

      {recommendations.svd_recommendations?.length > 0 && (
        <div className="bg-white rounded shadow p-4">
          <h3 className="font-medium mb-2">🧠 Collaborative Filtering (SVD)</h3>
          <ul className="list-disc ml-5 text-sm">
            {recommendations.svd_recommendations.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {recommendations.nn_recommendations?.length > 0 && (
        <div className="bg-white rounded shadow p-4">
          <h3 className="font-medium mb-2">🔗 Neural Network Model</h3>
          <ul className="list-disc ml-5 text-sm">
            {recommendations.nn_recommendations.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
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
