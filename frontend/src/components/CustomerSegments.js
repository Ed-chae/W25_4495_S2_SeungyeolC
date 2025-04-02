import React, { useEffect, useState } from "react";
import api from "../services/api";
import { motion } from "framer-motion";

function CustomerSegments() {
  const [segments, setSegments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .get("/customer-segmentation/")
      .then((res) => {
        setSegments(res.data.summary || []);
      })
      .catch((err) =>
        console.error("Error fetching customer segments:", err)
      )
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="text-gray-500">Loading customer segments...</p>;

  return (
    <motion.div
      className="p-4"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="text-xl font-semibold text-blue-600 mb-4">👥 Customer Segments</h2>
      <p className="text-sm text-gray-600 mb-4">
        Based on average revenue and purchase count. Each cluster is labeled by customer behavior.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {segments.map((seg, idx) => (
          <motion.div
            key={idx}
            className="bg-white p-4 rounded shadow"
            whileHover={{ scale: 1.03 }}
            transition={{ duration: 0.3 }}
          >
            <h3 className="text-lg font-semibold mb-2">
              🧩 Cluster #{seg.cluster_id}: {seg.label}
            </h3>
            <p className="text-sm text-gray-700">
              <strong>📊 Avg Revenue:</strong> ${seg.avg_revenue}
            </p>
            <p className="text-sm text-gray-700">
              <strong>🛒 Avg Purchases:</strong> {seg.avg_purchase_count}
            </p>
            <p className="text-sm text-gray-700">
              <strong>👥 Customers:</strong> {seg.total_customers}
            </p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

export default CustomerSegments;
