// src/components/MenuCategoryChart.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { motion } from "framer-motion";

ChartJS.register(ArcElement, Tooltip, Legend);

const MenuCategoryChart = () => {
  const [categoryData, setCategoryData] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/menu-category/")
      .then((res) => {
        setCategoryData(res.data.categories || {});
      })
      .catch((err) => {
        console.error("Failed to load menu category data.", err);
        setError("❌ Failed to load menu category data.");
      });
  }, []);

  if (error) return <div className="text-red-500">{error}</div>;
  if (!categoryData || Object.keys(categoryData).length === 0)
    return <p className="text-gray-500">Loading menu category charts...</p>;

  return (
    <motion.div
      className="dashboard-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="dashboard-card">
        🍽️ Menu Category Breakdown
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.entries(categoryData).map(([category, items], idx) => {
          const sortedItems = Object.entries(items).sort((a, b) => b[1] - a[1]);
          const labels = sortedItems.map(([item]) => item);
          const values = sortedItems.map(([, count]) => count);
          const topItem = labels[0];

          const data = {
            labels,
            datasets: [
              {
                label: `${category} Items`,
                data: values,
                backgroundColor: [
                  "#4ade80", "#60a5fa", "#facc15", "#f472b6", "#c084fc",
                  "#f87171", "#34d399", "#fcd34d", "#fca5a5", "#a78bfa"
                ],
              },
            ],
          };

          const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: "bottom" },
            },
          };

          return (
            <motion.div
              key={idx}
              className="card"
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.3 }}
            >
              <h3 className="text-md font-semibold text-center text-gray-800 mb-1">
                {category}
              </h3>
              <p className="card">
                🥇 Top item: <strong>{topItem}</strong>
              </p>
              <div className="card">
                <Pie data={data} options={options} />
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
};

export default MenuCategoryChart;
