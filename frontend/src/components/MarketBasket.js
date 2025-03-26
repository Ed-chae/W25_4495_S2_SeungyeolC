import React, { useEffect, useState } from "react";
import axios from "../services/api";

const MarketBasket = () => {
    const [marketData, setMarketData] = useState(null);

    useEffect(() => {
        axios.get("/market-basket/")
            .then(response => setMarketData(response.data))
            .catch(error => console.error("Error fetching market basket data:", error));
    }, []);

    if (!marketData) return <p>Loading Market Basket Analysis...</p>;

    return (
        <div>
            <h3>Market Basket Analysis</h3>
            <h4>Frequent Itemsets</h4>
            <ul>
                {marketData.frequent_itemsets.map((itemset, index) => (
                    <li key={index}>
                        <b>Items:</b> {itemset.itemsets.join(", ")} | <b>Support:</b> {itemset.support.toFixed(2)}
                    </li>
                ))}
            </ul>

            <h4>Association Rules</h4>
            <ul>
                {marketData.association_rules.map((rule, index) => (
                    <li key={index}>
                        <b>{rule.antecedents.join(", ")} → {rule.consequents.join(", ")}</b>
                        {" | Support: " + rule.support.toFixed(2) + " | Confidence: " + rule.confidence.toFixed(2) + " | Lift: " + rule.lift.toFixed(2)}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default MarketBasket;
