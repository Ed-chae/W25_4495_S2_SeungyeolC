// src/components/SentimentChart.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";

const SentimentChart = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/sentiment-results/")
      .then((res) => setData(res.data))
      .catch((err) => {
        console.error("❌ Sentiment error:", err);
        setError("Failed to fetch sentiment data.");
      });
  }, []);

  if (error) return <div className="text-red-500">{error}</div>;
  if (!data) return <div>Loading sentiment analysis...</div>;

  return (
    <div className="p-4 bg-white rounded shadow">
      <h2 className="text-xl font-semibold mb-2">🧠 Customer Sentiment Analysis</h2>
      <p className="mb-2 text-green-700">✅ Best item: <strong>{data.best_item}</strong></p>
      <p className="mb-4 text-red-600">⚠️ Worst item: <strong>{data.worst_item}</strong></p>

      <h3 className="font-semibold mb-2">📋 Sentiment Summary</h3>
      <table className="table-auto w-full text-sm border">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-2 py-1 text-left border">Item</th>
            <th className="px-2 py-1 text-left border">Positive %</th>
            <th className="px-2 py-1 text-left border">Negative %</th>
            <th className="px-2 py-1 text-left border">Positive</th>
            <th className="px-2 py-1 text-left border">Negative</th>
          </tr>
        </thead>
        <tbody>
          {data.summary.map((item, i) => (
            <tr key={i} className="border-t hover:bg-gray-50">
              <td className="px-2 py-1">{item.item}</td>
              <td className="px-2 py-1 text-green-700">{item.positive_pct}%</td>
              <td className="px-2 py-1 text-red-600">{item.negative_pct}%</td>
              <td className="px-2 py-1">{item.positive}</td>
              <td className="px-2 py-1">{item.negative}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default SentimentChart;
