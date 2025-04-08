// src/components/SentimentChart.js
import React, { useEffect, useState } from "react";
import api from "../services/api";
import { motion } from "framer-motion";

function SentimentChart() {
  const [summaryData, setSummaryData] = useState([]);
  const [bestItem, setBestItem] = useState(null);
  const [worstItem, setWorstItem] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/sentiment-results/")
      .then((res) => {
        const summary = Array.isArray(res.data) ? res.data : res.data?.summary;

        if (Array.isArray(summary) && summary.length > 0) {
          const processed = summary.map((item) => {
            const total = item.positive + item.negative;
            const positive_pct = total ? ((item.positive / total) * 100).toFixed(1) : 0;
            const negative_pct = total ? ((item.negative / total) * 100).toFixed(1) : 0;

            return {
              ...item,
              total_reviews: total,
              positive_pct: parseFloat(positive_pct),
              negative_pct: parseFloat(negative_pct),
            };
          });

          const sorted = [...processed].sort((a, b) => b.positive_pct - a.positive_pct);
          setSummaryData(sorted);

          const best = sorted[0];
          const worst = sorted.reduce((prev, curr) =>
            curr.negative_pct > prev.negative_pct ? curr : prev
          );

          setBestItem(`${best.item} (${best.positive_pct}% positive)`);
          setWorstItem(`${worst.item} (${worst.negative_pct}% negative)`);
        } else {
          setSummaryData([]);
        }
      })
      .catch((err) => {
        console.error("Error fetching sentiment results:", err);
        setError("❌ Failed to load sentiment summary.");
      });
  }, []);

  return (
    <motion.div
      className="bg-white rounded-xl shadow-md p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-2xl font-semibold text-indigo-700 mb-4">
        🧠 Customer Sentiment Summary
      </h2>

      {error && <p className="text-red-500">{error}</p>}

      {!error && summaryData.length === 0 ? (
        <p className="text-gray-500">No sentiment summary available.</p>
      ) : summaryData.length > 0 ? (
        <div className="space-y-4">
          <div className="text-sm">
            {bestItem && (
              <p className="text-green-600">
                ✅ <strong>Best item:</strong> {bestItem}
              </p>
            )}
            {worstItem && (
              <p className="text-red-500">
                ⚠️ <strong>Worst item:</strong> {worstItem}
              </p>
            )}
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full border border-gray-200 text-sm rounded-md overflow-hidden">
              <thead className="bg-gray-100 text-gray-700">
                <tr>
                  <th className="px-4 py-2 border">Item</th>
                  <th className="px-4 py-2 border text-green-600">👍 Positive %</th>
                  <th className="px-4 py-2 border text-red-600">👎 Negative %</th>
                  <th className="px-4 py-2 border">🧾 Total Reviews</th>
                </tr>
              </thead>
              <tbody>
                {summaryData.map((row, i) => (
                  <tr
                    key={i}
                    className="text-center hover:bg-gray-50 transition duration-200"
                  >
                    <td className="px-4 py-2 border font-medium">{row.item}</td>
                    <td className="px-4 py-2 border text-green-600 font-semibold">
                      {row.positive_pct}%
                    </td>
                    <td className="px-4 py-2 border text-red-600 font-semibold">
                      {row.negative_pct}%
                    </td>
                    <td className="px-4 py-2 border">{row.total_reviews}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </motion.div>
  );
}

export default SentimentChart;
