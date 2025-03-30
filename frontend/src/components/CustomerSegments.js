import React, { useEffect, useState } from "react";
import api from "../services/api";

function CustomerSegments() {
  const [summary, setSummary] = useState([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    api
      .get("/customer-segmentation/")
      .then((res) => {
        if (res.data.message) {
          setMessage(res.data.message);
        } else {
          setSummary(res.data.summary || []);
        }
      })
      .catch((err) => {
        console.error("Error fetching customer segments:", err);
        setMessage("❌ Failed to load segmentation results.");
      });
  }, []);

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">👥 Customer Segments</h2>

      {message && <p className="text-gray-600">{message}</p>}

      {summary.length > 0 ? (
        <table className="min-w-full border border-gray-300 text-sm bg-white shadow rounded">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 border">Cluster</th>
              <th className="px-4 py-2 border">Label</th>
              <th className="px-4 py-2 border">Avg Revenue</th>
              <th className="px-4 py-2 border">Avg Purchases</th>
              <th className="px-4 py-2 border"># of Customers</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((seg, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-4 py-2 border text-center">{seg.cluster_id}</td>
                <td className="px-4 py-2 border">{seg.label}</td>
                <td className="px-4 py-2 border">${seg.avg_revenue}</td>
                <td className="px-4 py-2 border">{seg.avg_purchase_count}</td>
                <td className="px-4 py-2 border">{seg.total_customers}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        !message && <p className="text-gray-600">No segmentation summary found.</p>
      )}
    </div>
  );
}

export default CustomerSegments;
