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


function dataToTraces(data, ind) {
    const traces = methods.map((method, index) => {
        return {
            xaxis: "x" + ind,
            yaxis: "y" + ind,
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
    return traces;
}

const singleScatterPlot = (id, data) => {
    let trace = dataToTraces(data);

    // Define chart layout
    const bgColor = 'rgb(33, 37, 41)';
    const axisConfig = {
        gridcolor: 'rgba(255, 255, 255, 0.2)',
        zerolinecolor: 'rgba(255, 255, 255, 0.4)',
        linecolor: 'rgba(255, 255, 255, 0.6)',
    };

    const layout = {
        plot_bgcolor: bgColor,
        paper_bgcolor: bgColor,
        font: { color: 'white' },
        xaxis: { 
            ...axisConfig,
            title: "Computation Time (mins)", 
            type: 'log',
        },
        yaxis: { 
            ...axisConfig,
            title: "Correlation", 
        },
        legend: {
            x: 1.02,
            y: 0.5,
            xanchor: 'left',
            yanchor: 'middle',
            orientation: 'v'
        },
        autosize: true
    };

    // Draw the chart
    Plotly.newPlot(id, trace, layout, { responsive: true });
    return layout;
};

// Update the legend position when the window is resized
// window.addEventListener('resize', () => updateLegendPosition(layout));
window.addEventListener('load', () => {
    singleScatterPlot('scatterplot-container1', table1);
});

const carousel = document.getElementById('trak-carousel');
let isMouseDown = false;
let startX;
let scrollLeft;

carousel.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    carousel.style.cursor = 'grabbing';
    startX = e.pageX - carousel.offsetLeft;
    scrollLeft = carousel.scrollLeft;
});

carousel.addEventListener('mouseleave', () => {
    isMouseDown = false;
    carousel.style.cursor = 'pointer';
});

carousel.addEventListener('mouseup', () => {
    isMouseDown = false;
    carousel.style.cursor = 'pointer';
});

carousel.addEventListener('mousemove', (e) => {
    if (!isMouseDown) return;
    e.preventDefault();
    const x = e.pageX - carousel.offsetLeft;
    const scrollX = (x - startX) * 2;
    carousel.scrollLeft = scrollLeft - scrollX;
});