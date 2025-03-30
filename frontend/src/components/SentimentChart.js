import React, { useEffect, useState } from "react";
import api from "../services/api";

function SentimentChart() {
  const [summaryData, setSummaryData] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api.get("/sentiment-results/")
      .then(res => {
        setSummaryData(res.data);
      })
      .catch(err => {
        console.error("Error fetching sentiment summary:", err);
        setError("❌ Failed to load sentiment summary.");
      });
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">📝 Customer Sentiment Summary</h2>
      {error && <p className="text-red-600">{error}</p>}

      {summaryData.length > 0 ? (
        <table className="min-w-full border border-gray-300 text-sm">
          <thead>
            <tr className="bg-gray-100">
              <th className="p-2 border">Item</th>
              <th className="p-2 border">👍 Positive</th>
              <th className="p-2 border">👎 Negative</th>
              <th className="p-2 border">Summary</th>
            </tr>
          </thead>
          <tbody>
            {summaryData.map((entry, idx) => (
              <tr key={idx}>
                <td className="p-2 border">{entry.item}</td>
                <td className="p-2 border">{entry.positive}</td>
                <td className="p-2 border">{entry.negative}</td>
                <td className="p-2 border">{entry.summary}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-gray-600">No sentiment data available.</p>
      )}
    </div>
  );
}

export default SentimentChart;
