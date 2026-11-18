/**
 * Smart Irrigation System - Frontend Dashboard & Prediction Logic
 * Connects to FastAPI REST endpoints, renders Chart.js analytics, and manages live inference.
 */

document.addEventListener("DOMContentLoaded", () => {
    loadDatasetStats();
    loadFeatureStatistics();
    loadModelBenchmarks();
    setupPredictionForm();
});

// Chart instance references to avoid duplicate canvas contexts
let pumpChartInstance = null;
let featuresChartInstance = null;
let modelsChartInstance = null;


// --------------------------------------------------------------------------
// 1. Dataset Overview & Pump Distribution Chart
// --------------------------------------------------------------------------
async function loadDatasetStats() {
    try {
        const response = await fetch("/api/dataset");
        if (!response.ok) {
            throw new Error(`Failed to load dataset metadata (HTTP ${response.status})`);
        }

        const data = await response.json();

        // Populate Overview Card Values
        document.getElementById("total-records").textContent = data.total_records.toLocaleString();
        document.getElementById("total-columns").textContent = data.total_columns;
        document.getElementById("pump-off").textContent = data.pump_status.off.toLocaleString();
        document.getElementById("pump-on").textContent = data.pump_status.on.toLocaleString();

        // Percentages & Dates
        document.getElementById("pump-off-pct").textContent = `${data.pump_status.off_percentage}% of total`;
        document.getElementById("pump-on-pct").textContent = `${data.pump_status.on_percentage}% of total`;

        if (data.date_range && data.date_range.start && data.date_range.end) {
            const startDate = data.date_range.start.split(" ")[0];
            const endDate = data.date_range.end.split(" ")[0];
            document.getElementById("date-range").textContent = `${startDate} to ${endDate}`;
        }

        renderPumpChart(data.pump_status);

    } catch (error) {
        console.error("Error loading dataset statistics:", error);
    }
}

function renderPumpChart(pumpStatus) {
    const ctx = document.getElementById("pumpChart");
    if (!ctx) return;

    if (pumpChartInstance) {
        pumpChartInstance.destroy();
    }

    pumpChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Pump OFF (0)", "Pump ON (1)"],
            datasets: [{
                data: [pumpStatus.off, pumpStatus.on],
                backgroundColor: ["#285640", "#8cc9a6"],
                borderColor: "#ffffff",
                borderWidth: 2,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        boxWidth: 14,
                        font: { size: 12, weight: 600 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const val = context.raw;
                            const total = pumpStatus.off + pumpStatus.on;
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${context.label}: ${val.toLocaleString()} (${pct}%)`;
                        }
                    }
                }
            },
            cutout: "62%"
        }
    });
}


// --------------------------------------------------------------------------
// 2. Feature Statistics Chart (Min, Mean, Max)
// --------------------------------------------------------------------------
async function loadFeatureStatistics() {
    try {
        const response = await fetch("/api/statistics");
        if (!response.ok) {
            throw new Error(`Failed to load statistics (HTTP ${response.status})`);
        }

        const stats = await response.json();
        renderFeaturesChart(stats);

    } catch (error) {
        console.error("Error loading feature statistics:", error);
    }
}

function renderFeaturesChart(stats) {
    const ctx = document.getElementById("featuresChart");
    if (!ctx) return;

    if (featuresChartInstance) {
        featuresChartInstance.destroy();
    }

    // Normalizing feature statistics for visual readability
    const labels = ["Soil Res. (Ω / 10)", "Temp. (°C)", "Humidity (%)"];
    const res = stats["Soil Moisture Resistance (Ohm)"] || {};
    const temp = stats["Ambient Temperature (deg. C)"] || {};
    const hum = stats["Atmospheric Humidity (%)"] || {};

    const minVals = [(res.min / 10) || 0, temp.min || 0, hum.min || 0];
    const meanVals = [(res.mean / 10) || 0, temp.mean || 0, hum.mean || 0];
    const maxVals = [(res.max / 10) || 0, temp.max || 0, hum.max || 0];

    featuresChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Min",
                    data: minVals,
                    backgroundColor: "#cbdcd0",
                    borderRadius: 4
                },
                {
                    label: "Mean",
                    data: meanVals,
                    backgroundColor: "#285640",
                    borderRadius: 4
                },
                {
                    label: "Max",
                    data: maxVals,
                    backgroundColor: "#8cc9a6",
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: "rgba(0, 0, 0, 0.05)" }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 12, font: { size: 12 } }
                },
                tooltip: {
                    callbacks: {
                        afterBody: function (items) {
                            if (items[0].label.includes("Soil Res")) {
                                return "(Values scaled by 1/10 for visualization)";
                            }
                            return "";
                        }
                    }
                }
            }
        }
    });
}


// --------------------------------------------------------------------------
// 3. Model Evaluation Benchmarks (Table & Comparative Chart)
// --------------------------------------------------------------------------
async function loadModelBenchmarks() {
    const tableBody = document.getElementById("models-table-body");
    try {
        const response = await fetch("/api/models");
        if (!response.ok) {
            throw new Error(`Failed to load models (HTTP ${response.status})`);
        }

        const data = await response.json();
        const models = data.models || [];

        // 1. Populate Table
        tableBody.innerHTML = "";
        models.forEach((model) => {
            const row = document.createElement("tr");
            if (model.Model === "Decision Tree" || model.Model === "Random Forest") {
                row.classList.add("best-row");
            }

            const cm = `[TN:${model.True_Negatives}, FP:${model.False_Positives}, FN:${model.False_Negatives}, TP:${model.True_Positives}]`;

            row.innerHTML = `
                <td><strong>${model.Model}</strong> ${model.Model === "Decision Tree" ? '<span style="color:#285640; font-size:11px;">(BEST / RULE)</span>' : ""}</td>
                <td>${(model.Accuracy * 100).toFixed(2)}%</td>
                <td>${(model.Precision * 100).toFixed(2)}%</td>
                <td>${(model.Recall * 100).toFixed(2)}%</td>
                <td>${(model.F1_Score * 100).toFixed(2)}%</td>
                <td>${(model.ROC_AUC * 100).toFixed(2)}%</td>
                <td style="font-family: monospace; font-size: 11px;">${cm}</td>
            `;
            tableBody.appendChild(row);
        });

        // 2. Render Chart
        renderModelsChart(models);

    } catch (error) {
        console.error("Error loading model benchmarks:", error);
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="loading-cell" style="color: #c82333;">
                    Failed to load model benchmarks: ${error.message}
                </td>
            </tr>
        `;
    }
}

function renderModelsChart(models) {
    const ctx = document.getElementById("modelsChart");
    if (!ctx) return;

    if (modelsChartInstance) {
        modelsChartInstance.destroy();
    }

    const labels = models.map((m) => m.Model);
    const accuracies = models.map((m) => +(m.Accuracy * 100).toFixed(2));
    const f1Scores = models.map((m) => +(m.F1_Score * 100).toFixed(2));
    const rocAucs = models.map((m) => +(m.ROC_AUC * 100).toFixed(2));

    modelsChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Accuracy (%)",
                    data: accuracies,
                    backgroundColor: "#102d24",
                    borderRadius: 4
                },
                {
                    label: "F1-Score (%)",
                    data: f1Scores,
                    backgroundColor: "#285640",
                    borderRadius: 4
                },
                {
                    label: "ROC-AUC (%)",
                    data: rocAucs,
                    backgroundColor: "#8cc9a6",
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 50,
                    max: 105,
                    title: { display: true, text: "Score Percentage (%)" },
                    grid: { color: "rgba(0, 0, 0, 0.05)" }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 14, font: { size: 12, weight: 600 } }
                }
            }
        }
    });
}


// --------------------------------------------------------------------------
// 4. Live Prediction Form Submission & Interaction
// --------------------------------------------------------------------------
function setupPredictionForm() {
    const form = document.getElementById("prediction-form");
    if (!form) return;

    const btn = document.getElementById("btn-predict");
    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");

    const placeholder = document.getElementById("prediction-placeholder");
    const resultBox = document.getElementById("prediction-result");
    const errorBox = document.getElementById("prediction-error");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // 1. Extract and validate input values
        const soilResistance = parseFloat(document.getElementById("soil_resistance").value);
        const temperature = parseFloat(document.getElementById("temperature").value);
        const humidity = parseFloat(document.getElementById("humidity").value);

        // 2. Set loading state
        btn.disabled = true;
        btnText.textContent = "Computing Inference...";
        btnSpinner.classList.remove("hidden");
        errorBox.classList.add("hidden");

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    soil_moisture_resistance: soilResistance,
                    ambient_temperature: temperature,
                    atmospheric_humidity: humidity
                })
            });

            const data = await response.json();

            if (!response.ok) {
                const detailMsg = Array.isArray(data.detail)
                    ? data.detail.map((d) => d.msg).join(", ")
                    : (data.detail || "Prediction request failed.");
                throw new Error(detailMsg);
            }

            // 3. Render Prediction Result
            placeholder.classList.add("hidden");
            errorBox.classList.add("hidden");
            resultBox.classList.remove("hidden");

            const badge = document.getElementById("status-badge");
            if (data.predicted_status === 1) {
                badge.className = "status-badge on";
                badge.textContent = "PUMP ON (1)";
            } else {
                badge.className = "status-badge off";
                badge.textContent = "PUMP OFF (0)";
            }

            document.getElementById("res-class").textContent = `${data.predicted_status} (${data.pump_status})`;
            document.getElementById("res-model").textContent = data.model_used;
            document.getElementById("res-res").textContent = `${data.input_features["Soil Moisture Resistance (Ohm)"]} Ω`;
            document.getElementById("res-temp").textContent = `${data.input_features["Ambient Temperature (deg. C)"]} °C`;
            document.getElementById("res-hum").textContent = `${data.input_features["Atmospheric Humidity (%)"]} %`;
            document.getElementById("res-note").textContent = data.note;

        } catch (err) {
            console.error("Prediction error:", err);
            placeholder.classList.add("hidden");
            resultBox.classList.add("hidden");
            errorBox.classList.remove("hidden");
            errorBox.textContent = `Inference Failed: ${err.message}`;
        } finally {
            // Restore button state
            btn.disabled = false;
            btnText.textContent = "Predict Pump Status";
            btnSpinner.classList.add("hidden");
        }
    });
}
