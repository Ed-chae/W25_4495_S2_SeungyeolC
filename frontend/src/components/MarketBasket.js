import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { motion } from "framer-motion";

const MarketBasket = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/market-basket/")
      .then((res) => {
        setData(res.data);
        setError("");
      })
      .catch((err) => {
        console.error("Error fetching market basket data:", err);
        setError("❌ Failed to load market basket analysis.");
      });
  }, []);

  if (error) return <p className="text-red-500">{error}</p>;
  if (!data) return <p className="text-gray-500">Loading market basket analysis...</p>;

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-semibold text-blue-800">🛒 Market Basket Analysis</h2>

      <div className="bg-white shadow rounded p-4">
        <h3 className="text-lg font-medium mb-2">🔁 Association Rules</h3>
        {data.length === 0 ? (
          <p className="text-gray-500">No association rules found.</p>
        ) : (
          <ul className="list-disc ml-5 space-y-1 text-sm">
            {data.map((rule, index) => (
              <li key={index}>
                <b>{Array.from(rule.antecedents).join(", ")}</b> ➡️{" "}
                <b>{Array.from(rule.consequents).join(", ")}</b> | Support:{" "}
                {rule.support.toFixed(2)}, Confidence: {rule.confidence.toFixed(2)}, Lift:{" "}
                {rule.lift.toFixed(2)}
              </li>
            ))}
          </ul>
        )}
      </div>
    </motion.div>
  );
};

export default MarketBasket;
