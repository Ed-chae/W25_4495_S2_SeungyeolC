import api from "./api";

export const resetSalesData = () => api.delete("/reset-sales-data/");
export const resetRestaurantOrders = () => api.delete("/reset-restaurant-orders/");
