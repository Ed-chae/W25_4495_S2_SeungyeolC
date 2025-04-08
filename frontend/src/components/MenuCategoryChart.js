// src/components/MenuCategoryChart.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

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
        setError("Failed to load menu category data.");
      });
  }, []);

  if (error) return <div className="text-red-500">{error}</div>;
  if (!categoryData || Object.keys(categoryData).length === 0)
    return <p>Loading menu category charts...</p>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-6">🍽️ Menu Category Breakdown</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.entries(categoryData).map(([category, items], idx) => {
          // Sort items within the category by sold count
          const sortedItems = Object.entries(items)
            .sort((a, b) => b[1] - a[1]); // Descending order

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
                  "#4ade80", "#60a5fa", "#facc15", "#f472b6", "#c084fc", "#f87171",
                  "#34d399", "#fcd34d", "#fca5a5", "#a78bfa",
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
            <div
              key={idx}
              className="bg-white rounded-xl shadow p-4 h-[430px] flex flex-col"
            >
              <h3 className="text-md font-semibold text-center mb-2">{category}</h3>
              <p className="text-sm text-center text-green-700 mb-2">
                🥇 Top item: <strong>{topItem}</strong>
              </p>
              <div className="flex-grow">
                <Pie data={data} options={options} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default MenuCategoryChart;
