import React, { useEffect, useState } from "react";
import api from "../services/api";

function SentimentChart() {
  const [details, setDetails] = useState([]);
  const [summary, setSummary] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/sentiment-results/")
      .then((res) => {
        console.log("Sentiment API response:", res.data);

        if (!res.data || !Array.isArray(res.data.details)) {
          setError("⚠️ Unexpected response format from API.");
          return;
        }

        setDetails(res.data.details);
        setSummary(res.data.summary || []);
        setError("");
      })
      .catch((err) => {
        console.error("Error fetching sentiment results:", err);
        setError("❌ Failed to fetch sentiment results.");
      });
  }, []);

  return (
    <div className="p-4 bg-white shadow-md rounded mb-6">
      <h2 className="text-xl font-semibold mb-4">🧠 Customer Sentiment Summary</h2>

      {error && <div className="text-red-600 mb-4">{error}</div>}

      {!error && details.length === 0 ? (
        <p className="text-gray-600">No sentiment data available. Upload a file to get started.</p>
      ) : (
        <>
          <table className="min-w-full border border-gray-300 text-sm">
            <thead>
              <tr className="bg-gray-100">
                <th className="p-2 border">Item</th>
                <th className="p-2 border">Review</th>
                <th className="p-2 border">Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {details.map((entry, idx) => (
                <tr key={idx}>
                  <td className="p-2 border">{entry.item}</td>
                  <td className="p-2 border">{entry.review}</td>
                  <td className="p-2 border">{entry.sentiment}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3 className="mt-6 text-lg font-medium">📊 Summary</h3>
          <ul className="list-disc ml-6 mt-2">
            {summary.map((s, idx) => (
              <li key={idx}>
                <strong>{s.item}</strong> – {s.summary}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default SentimentChart;
