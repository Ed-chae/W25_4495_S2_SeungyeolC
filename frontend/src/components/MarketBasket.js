// src/components/MarketBasket.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { motion } from "framer-motion";

const MarketBasket = () => {
  const [rules, setRules] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/market-basket/")
      .then((res) => {
        setRules(res.data.results || []);
      })
      .catch((err) => {
        console.error("❌ Error fetching market basket data:", err);
        setError("❌ Failed to load market basket analysis.");
      });
  }, []);

  const getTopRules = (metric) => {
    return [...rules]
      .sort((a, b) => b[metric] - a[metric])
      .slice(0, 3);
  };

  const formatRule = (rule, color) => (
    <motion.div
      key={`${rule.antecedents}-${rule.consequents}`}
      className={`mb-3 p-3 rounded-lg border shadow-sm bg-${color}-50 border-${color}-200`}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.3 }}
    >
      <div className="font-semibold text-gray-800">
        {rule.antecedents.join(", ")} → {rule.consequents.join(", ")}
      </div>
      <div className="text-sm text-gray-600 mt-1">
        Support: {rule.support.toFixed(2)} | Confidence: {rule.confidence.toFixed(2)} | Lift: {rule.lift.toFixed(2)}
      </div>
    </motion.div>
  );

  if (error) return <div className="text-red-500">{error}</div>;
  if (!rules.length) return <p>Loading Market Basket Analysis...</p>;

  return (
    <motion.div
      className="dashboard-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="text-xl font-bold text-indigo-700 mb-4">🛒 Market Basket Insights</h2>

      <div className="mb-6 text-sm text-gray-700">
        <p><strong>📘 Support:</strong> How often items are bought together in all orders.</p>
        <p><strong>📘 Confidence:</strong> Likelihood of buying the second item when buying the first.</p>
        <p><strong>📘 Lift:</strong> How much more likely the items are bought together than by chance.</p>
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-semibold text-blue-600 mb-2">⭐ Top 3 by Support</h3>
        {getTopRules("support").map((rule) => formatRule(rule, "blue"))}
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-semibold text-green-600 mb-2">🚀 Top 3 by Confidence</h3>
        {getTopRules("confidence").map((rule) => formatRule(rule, "green"))}
      </div>

      <div>
        <h3 className="text-lg font-semibold text-purple-600 mb-2">📈 Top 3 by Lift</h3>
        {getTopRules("lift").map((rule) => formatRule(rule, "purple"))}
      </div>
    </motion.div>
  );
};

export default MarketBasket;
