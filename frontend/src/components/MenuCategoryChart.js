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
        setCategoryData(res.data.categories);
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
      <h2 className="text-xl font-semibold mb-4">🍽️ Menu Category Breakdown</h2>
      {Object.entries(categoryData).map(([category, items], idx) => {
        const labels = Object.keys(items);
        const values = Object.values(items);

        const data = {
          labels,
          datasets: [
            {
              label: `${category} Items`,
              data: values,
              backgroundColor: [
                "#4ade80", "#60a5fa", "#facc15", "#f472b6", "#c084fc", "#f87171"
              ]
            }
          ]
        };

        return (
          <div key={idx} className="mb-8">
            <h3 className="text-lg font-semibold mb-2">{category}</h3>
            <Pie data={data} />
          </div>
        );
      })}
    </div>
  );
};

export default MenuCategoryChart;
