document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const statusMessage = document.getElementById("statusMessage");
    const uploadButton = document.getElementById("uploadButton");

    if (!fileInput || !statusMessage || !uploadButton) {
        console.error("Required elements not found on the page.");
        return;
    }

    uploadButton.addEventListener("click", uploadExcel);
});

async function uploadExcel() {
    const fileInput = document.getElementById("fileInput");
    const statusMessage = document.getElementById("statusMessage");

    // Ensure the status message element exists
    if (!statusMessage) {
        console.error("Element with ID 'statusMessage' not found.");
        return;
    }

    // Validate if file is selected
    if (!fileInput.files.length) {
        statusMessage.innerText = "❌ Please select a file to upload!";
        statusMessage.style.color = "red";
        return;
    }

    const file = fileInput.files[0];
    const allowedExtensions = [".xlsx", ".xls"];

    // Validate file type
    if (!allowedExtensions.some(ext => file.name.endsWith(ext))) {
        statusMessage.innerText = "❌ Invalid file type. Please upload an Excel file (.xlsx or .xls)";
        statusMessage.style.color = "red";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    statusMessage.innerText = "⏳ Uploading file, please wait...";
    statusMessage.style.color = "blue";

    try {
        const response = await fetch("http://127.0.0.1:8000/api/data/upload-excel/", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            statusMessage.innerText = "✅ File uploaded successfully!";
            statusMessage.style.color = "green";
            fetchRevenueData();
            fetchPredictions();
        } else {
            statusMessage.innerText = "❌ Upload failed: " + (result.detail || JSON.stringify(result.errors));
            statusMessage.style.color = "red";
        }
    } catch (error) {
        console.error("Error uploading file:", error);
        statusMessage.innerText = "❌ An unexpected error occurred while uploading.";
        statusMessage.style.color = "red";
    }
}

async function fetchRevenueData() {
    const statusMessage = document.getElementById("statusMessage");
    try {
        let response = await fetch("http://127.0.0.1:8000/api/analytics/revenue-trends/");
        let data = await response.json();

        let tableBody = document.querySelector("#revenueTable tbody");
        tableBody.innerHTML = "";

        data.forEach(row => {
            let tr = document.createElement("tr");
            tr.innerHTML = `<td>${row.YearMonth}</td><td>${row.Revenue}</td>`;
            tableBody.appendChild(tr);
        });

    } catch (error) {
        console.error("Error fetching revenue data:", error);
        if (statusMessage) {
            statusMessage.innerText = "❌ Error fetching revenue data.";
            statusMessage.style.color = "red";
        }
    }
}

async function fetchPredictions() {
    const statusMessage = document.getElementById("statusMessage");
    try {
        let response = await fetch("http://127.0.0.1:8000/api/analytics/predict-revenue/");
        let data = await response.json();

        let tableBody = document.querySelector("#predictionTable tbody");
        tableBody.innerHTML = "";

        for (let month in data.future_predictions) {
            let tr = document.createElement("tr");
            tr.innerHTML = `<td>${month}</td><td>${data.future_predictions[month]}</td>`;
            tableBody.appendChild(tr);
        }

    } catch (error) {
        console.error("Error fetching predictions:", error);
        if (statusMessage) {
            statusMessage.innerText = "❌ Error fetching predictions.";
            statusMessage.style.color = "red";
        }
    }
}
