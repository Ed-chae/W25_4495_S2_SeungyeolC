import React, { useEffect, useState } from "react";
import axios from "../services/api";

const MarketBasket = () => {
  const [marketData, setMarketData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios.get("/market-basket/")
      .then((response) => setMarketData(response.data))
      .catch((error) => {
        console.error("Error fetching market basket data:", error);
        setError("❌ Failed to fetch market basket data.");
      });
  }, []);

  if (error) return <p className="text-red-500">{error}</p>;
  if (!marketData) return <p className="text-gray-500">📊 Loading Market Basket Analysis...</p>;

  if (marketData.length === 0) {
    return <p className="text-yellow-600">⚠️ No basket analysis available. Please upload sales data.</p>;
  }

  return (
    <div className="bg-white shadow p-4 mb-6 rounded">
      <h2 className="text-xl font-bold mb-4">🛒 Market Basket Analysis</h2>

      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">📌 Association Rules</h3>
        <ul className="space-y-2 list-disc ml-6">
          {marketData.map((rule, index) => (
            <li key={index}>
              <span className="font-medium">
                {Array.from(rule.antecedents).join(", ")} → {Array.from(rule.consequents).join(", ")}
              </span>
              <div className="text-sm text-gray-600">
                Support: {rule.support.toFixed(2)}, Confidence: {rule.confidence.toFixed(2)}, Lift: {rule.lift.toFixed(2)}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default MarketBasket;
