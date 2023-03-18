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
    { method: "Representation Dist.", computationTime: 50, correlation: 0.029, error: 0.0060 },
    { method: "GAS [HL22]", computationTime: 30, correlation: 0.047, error: 0.0072 },
    { method: "TracIn [PLS+20]", computationTime: 15, correlation: 0.056, error: 0.0073 },
    { method: "TracIn [PLS+20]", computationTime: 300, correlation: 0.055, error: 0.0070 },
  ];

  const table2 = [
    { method: "Datamodel [IPE+22]", computationTime: 43200.0, correlation: 0.178846, error: 0.033482 },
    { method: "Datamodel [IPE+22]", computationTime: 85500.0, correlation: 0.258097, error: 0.033210 },
    { method: "Datamodel [IPE+22]", computationTime: 175500.0, correlation: 0.344712, error: 0.031926 },
    { method: "TracIn [PLS+20]", computationTime: 284, correlation: 0.072531, error: 0.025567 },
    { method: "Emp. Influence [FZ20]", computationTime: 43200.0, correlation: 0.165813, error: 0.041610 },
    { method: "Emp. Influence [FZ20]", computationTime: 85500.0, correlation: 0.202983, error: 0.046267 },
    { method: "Emp. Influence [FZ20]", computationTime: 175500.0, correlation: 0.225179, error: 0.044004 },
    { method: "TRAK", computationTime: 64, correlation: 0.178444, error: 0.005628 },
    { method: "TRAK", computationTime: 640, correlation: 0.416264, error: 0.010370 },
    { method: "TRAK", computationTime: 6400, correlation: 0.593538, error: 0.014348 },
    { method: "IF [KL17]", computationTime: 18042.9, correlation: 0.113670, error: 0.043225 },
    { method: "IF [KL17]", computationTime: 90214.5, correlation: 0.155935, error: 0.029792 },
    { method: "Rep. Distance", computationTime: 90, correlation: 0.050549, error: 0.047340 },
    { method: "Rep. Distance", computationTime: 180, correlation: 0.050256, error: 0.048500 },
    { method: "GAS [HL22]", computationTime: 284.0, correlation: 0.077749, error: 0.028504 },
  ];


  const methods = [
    "Datamodel [IPE+22]",
    "TracIn [PLS+20]",
    "Emp. Influence [FZ20]",
    "TRAK",
    "IF [KL17]",
    "Rep. Distance",
    "GAS [HL22]",
  ];
  
  const colors = [
    "rgba(255, 0, 0, 0.8)",
    "rgba(0, 255, 0, 0.8)",
    "rgba(0, 0, 255, 0.8)",
    "rgba(255, 128, 0, 0.8)",
    "rgba(128, 0, 255, 0.8)",
    "rgba(0, 255, 255, 0.8)",
    "rgba(255, 0, 255, 0.8)",
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
    marker: { color: colors[index], size: 12 },
  };
});

// Define chart layout
// Log scale x axis
const layout = {
    title: "Scatterplot",
    xaxis: { title: "Computation Time (mins)" , type: 'log'},
    yaxis: { title: "Correlation" }
};

// Draw the chart
Plotly.newPlot(id, traces, layout);
  };
  
  scatterPlot(table1, 'scatterplot-container1');
  scatterPlot(table2, 'scatterplot-container2');
  
  