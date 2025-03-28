import React, { useEffect, useState } from "react";
import api from "../services/api";

function SentimentChart() {
  const [data, setData] = useState([]);
  const [summary, setSummary] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/sentiment-results/")
      .then((res) => {
        console.log("Sentiment API response:", res.data);

        if (!Array.isArray(res.data)) {
          setError("⚠️ Unexpected response format from API.");
          return;
        }

        setData(res.data);

        const sentimentMap = {};
        res.data.forEach(({ product, sentiment }) => {
          if (!sentimentMap[product]) {
            sentimentMap[product] = { positive: 0, negative: 0 };
          }
          if (sentiment === "POSITIVE") {
            sentimentMap[product].positive += 1;
          } else {
            sentimentMap[product].negative += 1;
          }
        });

        const summaryObj = {};
        Object.keys(sentimentMap).forEach((product) => {
          const total = sentimentMap[product].positive + sentimentMap[product].negative;
          const negativeRate = ((sentimentMap[product].negative / total) * 100).toFixed(1);
          summaryObj[product] = `${negativeRate}% negative`;
        });

        setSummary(summaryObj);
      })
      .catch((err) => {
        console.error("Error fetching sentiment results:", err);
        setError("❌ Failed to fetch sentiment results.");
      });
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">Customer Sentiment Summary</h2>

      {error && <div className="text-red-600 mb-4">{error}</div>}

      {Array.isArray(data) && data.length > 0 ? (
        <>
          <table className="min-w-full border border-gray-300 text-sm">
            <thead>
              <tr className="bg-gray-100">
                <th className="p-2 border">Product</th>
                <th className="p-2 border">Review</th>
                <th className="p-2 border">Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {data.map((entry, idx) => (
                <tr key={idx}>
                  <td className="p-2 border">{entry.product}</td>
                  <td className="p-2 border">{entry.review}</td>
                  <td className="p-2 border">{entry.sentiment}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3 className="mt-6 text-lg font-medium">Summary</h3>
          <ul className="list-disc ml-6 mt-2">
            {Object.entries(summary).map(([product, summaryText], idx) => (
              <li key={idx}>
                <strong>{product}</strong> – {summaryText}
              </li>
            ))}
          </ul>
        </>
      ) : (
        !error && <p className="text-gray-600">No sentiment data available.</p>
      )}
    </div>
  );
}

export default SentimentChart;
