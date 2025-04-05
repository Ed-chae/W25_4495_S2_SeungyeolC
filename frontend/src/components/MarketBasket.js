// src/components/MarketBasket.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";

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

  const formatRule = (rule) => (
    <li key={`${rule.antecedents}-${rule.consequents}`}>
      <strong>{rule.antecedents.join(", ")} → {rule.consequents.join(", ")}</strong>{" "}
      | Support: {rule.support.toFixed(2)} | Confidence: {rule.confidence.toFixed(2)} | Lift: {rule.lift.toFixed(2)}
    </li>
  );

  if (error) return <div className="text-red-500">{error}</div>;
  if (!rules.length) return <p>Loading Market Basket Analysis...</p>;

  return (
    <div className="p-4 bg-white shadow rounded">
      <h2 className="text-xl font-semibold mb-4">🛒 Market Basket Insights</h2>

      <div className="mb-4 text-sm text-gray-700">
        <p><strong>📘 Support:</strong> How often items are bought together in all orders.</p>
        <p><strong>📘 Confidence:</strong> Likelihood of buying the second item when buying the first.</p>
        <p><strong>📘 Lift:</strong> How much more likely the items are bought together than by chance.</p>
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">⭐ Top 3 by Support (most common pairs)</h3>
        <ul className="list-disc ml-6 space-y-1">
          {getTopRules("support").map(formatRule)}
        </ul>
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">🚀 Top 3 by Confidence (most predictable)</h3>
        <ul className="list-disc ml-6 space-y-1">
          {getTopRules("confidence").map(formatRule)}
        </ul>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-2">📈 Top 3 by Lift (strongest associations)</h3>
        <ul className="list-disc ml-6 space-y-1">
          {getTopRules("lift").map(formatRule)}
        </ul>
      </div>
    </div>
  );
};

export default MarketBasket;
