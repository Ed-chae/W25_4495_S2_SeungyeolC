import React, { useEffect, useState } from "react";
import api from "../services/api";

function CustomerSegments() {
  const [segments, setSegments] = useState([]);
  const [summary, setSummary] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .get("/customer-segmentation/")
      .then((res) => {
        setSegments(res.data.raw_data || []);
        setSummary(res.data.summary || []);
        setError(res.data.message || null);
      })
      .catch((err) => {
        console.error("Error fetching customer segments:", err);
        setError("Failed to fetch customer segments.");
      });
  }, []);

  if (error) {
    return <p className="text-red-500 p-4">{error}</p>;
  }

  if (segments.length === 0) {
    return <p className="p-4 text-gray-600">No segmentation data available. Please upload a file first.</p>;
  }

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Customer Segments</h2>
      <p className="text-sm text-gray-600 mb-4">
        Customers are grouped by spending and purchasing patterns. Group numbers represent KMeans clusters.
      </p>

      <div className="overflow-x-auto mb-6">
        <table className="min-w-full border border-gray-300 text-sm bg-white shadow rounded">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 border">Customer ID</th>
              <th className="px-4 py-2 border">Revenue</th>
              <th className="px-4 py-2 border">Purchase Count</th>
              <th className="px-4 py-2 border">KMeans Cluster</th>
              <th className="px-4 py-2 border">DBSCAN Cluster</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((seg, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-4 py-2 border">{seg.customer_id}</td>
                <td className="px-4 py-2 border">${seg.revenue.toFixed(2)}</td>
                <td className="px-4 py-2 border">{seg.purchase_count}</td>
                <td className="px-4 py-2 border">{seg.kmeans_cluster}</td>
                <td className="px-4 py-2 border">{seg.dbscan_cluster}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="text-lg font-semibold mb-2">Cluster Summary</h3>
      <ul className="list-disc ml-6 text-sm">
        {summary.map((s, idx) => (
          <li key={idx}>
            <strong>Cluster {s.cluster_id}:</strong> {s.label} — Avg Revenue: ${s.avg_revenue}, Avg Purchases: {s.avg_purchase_count}, Total Customers: {s.total_customers}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default CustomerSegments;
