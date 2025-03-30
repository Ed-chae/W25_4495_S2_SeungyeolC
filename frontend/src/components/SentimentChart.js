import React, { useEffect, useState } from "react";
import api from "../services/api";

function SentimentChart() {
  const [summary, setSummary] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/sentiment-results/")
      .then((res) => {
        if (!res.data || !res.data.summary) {
          setError("⚠️ Unexpected response format.");
          return;
        }

        const formatted = res.data.summary.map((item) => {
          const total = item.positive + item.negative;
          const negativeRate = (item.negative / total) * 100;
          return {
            item: item.item,
            positive: item.positive,
            negative: item.negative,
            summary:
              negativeRate < 50
                ? "✅ Positive"
                : "⚠️ Negative",
            negativeRate: negativeRate.toFixed(1) + "% negative"
          };
        });

        setSummary(formatted);
      })
      .catch((err) => {
        console.error("Error fetching sentiment summary:", err);
        setError("❌ Failed to load sentiment summary.");
      });
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">📊 Customer Sentiment Summary</h2>

      {error && <div className="text-red-600 mb-4">{error}</div>}

      {summary.length > 0 ? (
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
            {summary.map((item, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="p-2 border">{item.item}</td>
                <td className="p-2 border text-center">{item.positive}</td>
                <td className="p-2 border text-center">{item.negative}</td>
                <td className="p-2 border">
                  {item.summary} ({item.negativeRate})
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        !error && <p className="text-gray-600">No sentiment summary available.</p>
      )}
    </div>
  );
}

export default SentimentChart;
