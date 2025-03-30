import React, { useEffect, useState } from "react";
import axios from "../services/api";

const MarketBasket = () => {
  const [marketData, setMarketData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/market-basket/")
      .then((response) => {
        const data = response.data;

        if (!data || typeof data !== "object") {
          setError("❌ Invalid market basket response.");
        } else {
          setMarketData(data);
        }
      })
      .catch((error) => {
        console.error("Error fetching market basket data:", error);
        setError("❌ Failed to fetch market basket analysis.");
      });
  }, []);

  if (error) return <p className="text-red-600">{error}</p>;
  if (!marketData) return <p>Loading Market Basket Analysis...</p>;

  return (
    <div className="p-4">
      <h3 className="text-xl font-bold mb-4">🛒 Market Basket Analysis</h3>

      <h4 className="font-semibold">Frequent Itemsets</h4>
      <ul className="mb-4 list-disc ml-6">
        {marketData.frequent_itemsets?.map((itemset, index) => (
          <li key={index}>
            <strong>Items:</strong> {itemset.itemsets.join(", ")} | <strong>Support:</strong> {itemset.support.toFixed(2)}
          </li>
        ))}
      </ul>

      <h4 className="font-semibold">Association Rules</h4>
      <ul className="list-disc ml-6">
        {marketData.association_rules?.map((rule, index) => (
          <li key={index}>
            <strong>{rule.antecedents.join(", ")} → {rule.consequents.join(", ")}</strong>
            {" | Support: " + rule.support.toFixed(2)}
            {" | Confidence: " + rule.confidence.toFixed(2)}
            {" | Lift: " + rule.lift.toFixed(2)}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default MarketBasket;
