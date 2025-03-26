import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Scatter } from "react-chartjs-2";

const SalesAnomalies = () => {
    const [anomalyData, setAnomalyData] = useState({ datasets: [] });

    useEffect(() => {
        axios.get("/sales-anomalies/")
            .then(response => {
                const sales = response.data;

                const normalSales = sales.filter(s => s.is_anomaly === "Normal");
                const anomalies = sales.filter(s => s.is_anomaly === "Anomaly");

                setAnomalyData({
                    datasets: [
                        {
                            label: "Normal Sales",
                            data: normalSales.map(s => ({ x: s.date, y: s.revenue })),
                            pointRadius: 5,
                            borderColor: "#4CAF50"
                        },
                        {
                            label: "Anomalies",
                            data: anomalies.map(s => ({ x: s.date, y: s.revenue })),
                            pointRadius: 5,
                            borderColor: "#FF5733"
                        }
                    ]
                });
            })
            .catch(error => console.error("Error fetching sales anomalies:", error));
    }, []);

    return (
        <div>
            <h3>Sales Anomaly Detection</h3>
            <Scatter data={anomalyData} />
        </div>
    );
};

export default SalesAnomalies;
