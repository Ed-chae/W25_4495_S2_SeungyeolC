import React, { useEffect, useState } from "react";
import api from "../services/api";

function CustomerSegments() {
  const [segments, setSegments] = useState([]);

  useEffect(() => {
    api
      .get("/customer-segmentation/")
      .then((res) => setSegments(res.data))
      .catch((err) => console.error("Error fetching customer segments:", err));
  }, []);

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Customer Segments</h2>
      <p className="text-sm text-gray-600 mb-4">
        Customers are grouped by spending and purchasing patterns. Group numbers represent different segment clusters.
      </p>
      <div className="overflow-x-auto">
        <table className="min-w-full border border-gray-300 text-sm bg-white shadow rounded">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 border">Customer ID</th>
              <th className="px-4 py-2 border">Revenue</th>
              <th className="px-4 py-2 border">Purchase Count</th>
              <th className="px-4 py-2 border">Group</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((seg, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-4 py-2 border">{seg.customer_id}</td>
                <td className="px-4 py-2 border">${seg.revenue.toFixed(2)}</td>
                <td className="px-4 py-2 border">{seg.purchase_count}</td>
                <td className="px-4 py-2 border">{seg.cluster}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default CustomerSegments;
