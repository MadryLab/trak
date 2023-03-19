function updateLegendPosition(layout) {
    layout.legend = {
        x: 0.5,
        y: 1.2,
        xanchor: 'center',
        orientation: 'h',
    };

    Plotly.update('scatterplot-container1', {}, layout);
    Plotly.update('scatterplot-container2', {}, layout);
}

const table1 = [
    { method: "TRAK", computationTime: 1, correlation: 0.058, error: 0.0039 },
    { method: "TRAK", computationTime: 5, correlation: 0.12, error: 0.0036 },
    { method: "TRAK", computationTime: 20, correlation: 0.27, error: 0.0042 },
    { method: "TRAK", computationTime: 100, correlation: 0.40, error: 0.0036 },
    { method: "Datamodel [IPE+22]", computationTime: 500, correlation: 0.060, error: 0.0045 },
    { method: "Datamodel [IPE+22]", computationTime: 2500, correlation: 0.20, error: 0.0037 },
    { method: "Datamodel [IPE+22]", computationTime: 10000, correlation: 0.39, error: 0.0037 },
    { method: "Datamodel [IPE+22]", computationTime: 25000, correlation: 0.48, error: 0.0030 },
    { method: "Datamodel [IPE+22]", computationTime: 50000, correlation: 0.54, error: 0.0028 },
    { method: "Emp. Influence [FZ20]", computationTime: 500, correlation: 0.048, error: 0.0043 },
    { method: "Emp. Influence [FZ20]", computationTime: 2500, correlation: 0.10, error: 0.0045 },
    { method: "Emp. Influence [FZ20]", computationTime: 10000, correlation: 0.19, error: 0.0041 },
    { method: "Emp. Influence [FZ20]", computationTime: 25000, correlation: 0.28, error: 0.0041 },
    { method: "IF-Arnoldi [SZV+22]", computationTime: 120, correlation: 0.020, error: 0.0048 },
    { method: "IF [KL17]", computationTime: 25003, correlation: 0.037, error: 0.021 },
    { method: "Representation Distance", computationTime: 50, correlation: 0.029, error: 0.0060 },
    { method: "GAS [HL22]", computationTime: 30, correlation: 0.047, error: 0.0072 },
    { method: "TracIn [PLS+20]", computationTime: 15, correlation: 0.056, error: 0.0073 },
    { method: "TracIn [PLS+20]", computationTime: 300, correlation: 0.055, error: 0.0070 },
];

const table2 = [
    { method: "Datamodel [IPE+22]", computationTime: 43200.0, correlation: 0.18, error: 0.033 },
    { method: "Datamodel [IPE+22]", computationTime: 85500.0, correlation: 0.26, error: 0.033 },
    { method: "Datamodel [IPE+22]", computationTime: 175500.0, correlation: 0.34, error: 0.032 },
    { method: "TracIn [PLS+20]", computationTime: 284, correlation: 0.073, error: 0.026 },
    { method: "Emp. Influence [FZ20]", computationTime: 43200.0, correlation: 0.17, error: 0.042 },
    { method: "Emp. Influence [FZ20]", computationTime: 85500.0, correlation: 0.20, error: 0.046 },
    { method: "Emp. Influence [FZ20]", computationTime: 175500.0, correlation: 0.23, error: 0.044 },
    { method: "TRAK", computationTime: 64, correlation: 0.18, error: 0.0056 },
    { method: "TRAK", computationTime: 640, correlation: 0.42, error: 0.010 },
    { method: "TRAK", computationTime: 6400, correlation: 0.59, error: 0.014 },
    { method: "IF [KL17]", computationTime: 18042.9, correlation: 0.11, error: 0.043 },
    { method: "IF [KL17]", computationTime: 90214.5, correlation: 0.16, error: 0.030 },
    { method: "Representation Distance", computationTime: 90, correlation: 0.051, error: 0.047 },
    { method: "Representation Distance", computationTime: 180, correlation: 0.050, error: 0.049 },
    { method: "GAS [HL22]", computationTime: 284.0, correlation: 0.078, error: 0.029 },
];



const methods = [
    "Datamodel [IPE+22]",
    "TracIn [PLS+20]",
    "Emp. Influence [FZ20]",
    "TRAK",
    "IF [KL17]",
    "Representation Distance",
    "GAS [HL22]",
];

const colors = [
    '#FF6B6B', // "rgba(255, 0, 0, 0.8)",
    '#F0E66B', // "rgba(0, 255, 0, 0.8)",
    '#4EDEA4', // "rgba(0, 0, 255, 0.8)",
    '#5AB5FF', // "rgba(255, 128, 0, 0.8)",
    '#936BFF', // "rgba(128, 0, 255, 0.8)",
    '#FF8ECF', // "rgba(0, 255, 255, 0.8)",
    '#FFB86B' // "rgba(255, 0, 255, 0.8)",
];


const scatterPlot = (data, id) => {
    const traces = methods.map((method, index) => {
        return {
            x: data
                .filter((entry) => entry.method === method)
                .map((entry) => entry.computationTime),
            y: data
                .filter((entry) => entry.method === method)
                .map((entry) => entry.correlation),
            error_y: {
                type: "data",
                array: data
                    .filter((entry) => entry.method === method)
                    .map((entry) => entry.error),
                visible: true,
            },
            mode: "markers",
            type: "scatter",
            name: method,
            marker: { color: colors[index], size: 12 }
        };
    });

    // Define chart layout
    // Log scale x axis
    const bgColor = 'rgb(33, 37, 41)';
    const layout = {
        plot_bgcolor: bgColor,
        paper_bgcolor: bgColor,
        font: { color: 'white' },
        xaxis: { 
            title: "Computation Time (mins)", 
            type: 'log',
            gridcolor: 'rgba(255, 255, 255, 0.2)',
            zerolinecolor: 'rgba(255, 255, 255, 0.4)',
            linecolor: 'rgba(255, 255, 255, 0.6)',
        },
        yaxis: { 
            title: "Correlation", 
            gridcolor: 'rgba(255, 255, 255, 0.2)',
            zerolinecolor: 'rgba(255, 255, 255, 0.4)',
            linecolor: 'rgba(255, 255, 255, 0.6)',
        },
        // legend: { orientation: "h" },
        autosize: true
    };

    // Draw the chart
    Plotly.newPlot(id, traces, layout, { responsive: true });
    return layout;
};

// Update the legend position when the window is resized
// window.addEventListener('resize', () => updateLegendPosition(layout));
window.addEventListener('load', () => {
    scatterPlot(table1, 'scatterplot-container1');
    document.getElementById('scatterplot-container1');
    // .on('plotly_ready', () => {
        // updateLegendPosition(layout);
        // Plotly.Plots.resize('scatterplot-container1');
    // });
    scatterPlot(table2, 'scatterplot-container2');
    // updateLegendPosition(layout);
});