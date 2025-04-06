// src/components/MenuCategoryChart.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
ChartJS.register(ArcElement, Tooltip, Legend);

const MenuCategoryChart = () => {
  const [categories, setCategories] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/menu-categories/")
      .then((res) => {
        setCategories(res.data.categories);
      })
      .catch((err) => {
        console.error("❌ Error fetching menu categories:", err);
        setError("Failed to load menu category data.");
      });
  }, []);

  if (error) return <div className="text-red-500">{error}</div>;
  if (!Object.keys(categories).length) return <div>Loading...</div>;

  const data = {
    labels: Object.keys(categories),
    datasets: [
      {
        data: Object.values(categories),
        backgroundColor: [
          "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40"
        ]
      }
    ]
  };

  return (
    <div className="p-4 bg-white rounded shadow mt-6">
      <h2 className="text-xl font-semibold mb-4">🍽️ Menu Category Breakdown</h2>
      <Pie data={data} />
    </div>
  );
};

export default MenuCategoryChart;
