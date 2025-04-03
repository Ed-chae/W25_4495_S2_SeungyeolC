// src/components/MarketBasket.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";

const MarketBasket = () => {
  const [marketData, setMarketData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/market-basket/")
      .then((res) => {
        setMarketData(res.data);
      })
      .catch((err) => {
        console.error("❌ Error fetching market basket data:", err);
        setError("❌ Failed to load market basket analysis.");
      });
  }, []);

  if (error) return <div className="text-red-500">{error}</div>;
  if (!marketData) return <p>Loading Market Basket Analysis...</p>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">🛒 Market Basket Analysis</h2>

      <h3 className="text-lg font-medium mt-4 mb-2">📈 Association Rules</h3>
      {marketData.results && marketData.results.length > 0 ? (
        <ul className="list-disc ml-6 space-y-1">
          {marketData.results.map((rule, index) => (
            <li key={index}>
              <strong>{rule.antecedents?.join(", ")} → {rule.consequents?.join(", ")}</strong>{" "}
              | Support: {rule.support.toFixed(2)} | Confidence: {rule.confidence.toFixed(2)} | Lift: {rule.lift.toFixed(2)}
            </li>
          ))}
        </ul>
      ) : (
        <p>No association rules found.</p>
      )}
    </div>
  );
};

export default MarketBasket;
