import React, { useEffect, useState } from "react";
import api from "../services/api";
import { motion } from "framer-motion";

function SentimentChart() {
  const [summaryData, setSummaryData] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/sentiment-results/")
      .then((res) => {
        if (res.data?.summary) {
          const processed = res.data.summary.map((item) => {
            const positivity =
              item.negative / (item.positive + item.negative) < 0.5
                ? "😊 Positive"
                : "😟 Negative";

            return {
              item: item.item,
              positive: item.positive,
              negative: item.negative,
              summary: item.summary,
              mood: positivity,
            };
          });
          setSummaryData(processed);
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
      className="p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-bold mb-4 text-indigo-700">
        🧠 Customer Sentiment Summary
      </h2>

      {error && <p className="text-red-500">{error}</p>}

      {!error && summaryData.length === 0 ? (
        <p className="text-gray-500">No sentiment summary available.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full border border-gray-300 text-sm bg-white rounded shadow">
            <thead className="bg-gray-100">
              <tr>
                <th className="p-2 border">Item</th>
                <th className="p-2 border">👍 Positive</th>
                <th className="p-2 border">👎 Negative</th>
                <th className="p-2 border">Summary</th>
                <th className="p-2 border">Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {summaryData.map((row, idx) => (
                <tr key={idx} className="hover:bg-gray-50 text-center">
                  <td className="p-2 border">{row.item}</td>
                  <td className="p-2 border">{row.positive}</td>
                  <td className="p-2 border">{row.negative}</td>
                  <td className="p-2 border">{row.summary}</td>
                  <td
                    className={`p-2 border font-semibold ${
                      row.mood.includes("Positive")
                        ? "text-green-600"
                        : "text-red-500"
                    }`}
                  >
                    {row.mood}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </motion.div>
  );
}

export default SentimentChart;
