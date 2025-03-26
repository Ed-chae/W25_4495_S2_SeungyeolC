import React, { useEffect, useState } from "react";
import axios from "../services/api";

const Recommendations = ({ userId }) => {
    const [recommendations, setRecommendations] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!userId) {
            setError("No user ID provided.");
            setLoading(false);
            return;
        }

        axios.get(`/product-recommendations/?user_id=${userId}`)
            .then(response => {
                setRecommendations(response.data);
                setError(null);
            })
            .catch(error => {
                console.error("Error fetching recommendations:", error);
                setError("Failed to load recommendations. Please try again.");
            })
            .finally(() => setLoading(false));
    }, [userId]);

    if (loading) return <p>Loading recommendations...</p>;
    if (error) return <p style={{ color: "red" }}>{error}</p>;
    if (!recommendations) return <p>No recommendations available.</p>;

    return (
        <div>
            <h3>Personalized Product Recommendations</h3>

            {recommendations.svd_recommendations?.length > 0 ? (
                <>
                    <p><b>Based on Collaborative Filtering:</b></p>
                    <ul>
                        {recommendations.svd_recommendations.map((product, index) => (
                            <li key={index}>{product}</li>
                        ))}
                    </ul>
                </>
            ) : (
                <p>No collaborative filtering recommendations available.</p>
            )}

            {recommendations.nn_recommendations?.length > 0 ? (
                <>
                    <p><b>Based on Neural Network Model:</b></p>
                    <ul>
                        {recommendations.nn_recommendations.map((product, index) => (
                            <li key={index}>{product}</li>
                        ))}
                    </ul>
                </>
            ) : (
                <p>No neural network recommendations available.</p>
            )}
        </div>
    );
};

export default Recommendations;
