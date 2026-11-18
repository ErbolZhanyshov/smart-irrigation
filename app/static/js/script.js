async function loadDatasetStats() {
    try {
        const response = await fetch("/api/dataset");

        if (!response.ok) {
            throw new Error("Dataset could not be loaded");
        }

        const data = await response.json();

        document.getElementById("total-records").textContent =
            data.total_records;

        document.getElementById("total-columns").textContent =
            data.total_columns;

        document.getElementById("pump-off").textContent =
            data.pump_status.off;

        document.getElementById("pump-on").textContent =
            data.pump_status.on;

    } catch (error) {
        console.error("Error:", error);
    }
}

document.addEventListener("DOMContentLoaded", loadDatasetStats);

async function loadPumpChart() {
    try {
        const response = await fetch("/api/dataset");
        const data = await response.json();

        const ctx = document.getElementById("pumpChart");

        new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Pump OFF", "Pump ON"],
                datasets: [{
                    data: [
                        data.pump_status.off,
                        data.pump_status.on
                    ],
                    backgroundColor: [
                        "#285640",
                        "#8cc9a6"
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom"
                    }
                }
            }
        });

    } catch (error) {
        console.error("Chart error:", error);
    }
}

document.addEventListener("DOMContentLoaded", loadPumpChart);