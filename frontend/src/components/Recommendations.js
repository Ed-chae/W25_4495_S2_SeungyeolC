import React, { useEffect, useState } from "react";
import axios from "../services/api";

const Recommendations = ({ userId }) => {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!userId) {
      setError("❌ No user ID provided.");
      setLoading(false);
      return;
    }

    axios
      .get(`/product-recommendations/?user_id=${userId}`)
      .then((response) => {
        setRecommendations(response.data);
        setError(null);
      })
      .catch((error) => {
        console.error("Error fetching recommendations:", error);
        setError("❌ Failed to load recommendations. Please try again.");
      })
      .finally(() => setLoading(false));
  }, [userId]);

  if (loading) return <p className="text-gray-600">⏳ Loading recommendations...</p>;
  if (error) return <p className="text-red-500">{error}</p>;
  if (!recommendations) return <p>⚠️ No recommendations found.</p>;

  return (
    <div className="bg-white shadow p-4 mb-6 rounded">
      <h2 className="text-xl font-bold mb-4">🎯 Personalized Menu Recommendations</h2>

      {recommendations.svd_recommendations?.length > 0 ? (
        <div className="mb-4">
          <p className="font-semibold mb-2">📊 Based on Collaborative Filtering (SVD):</p>
          <ul className="list-disc list-inside text-gray-700">
            {recommendations.svd_recommendations.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="text-gray-500">No collaborative filtering recommendations available.</p>
      )}

      {recommendations.nn_recommendations?.length > 0 ? (
        <div className="mb-4">
          <p className="font-semibold mb-2">🧠 Based on Neural Network:</p>
          <ul className="list-disc list-inside text-gray-700">
            {recommendations.nn_recommendations.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="text-gray-500">No neural network recommendations available.</p>
      )}
    </div>
  );
};

export default Recommendations;
