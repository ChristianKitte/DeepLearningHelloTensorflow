function DrawGraph(divID, spannung, strom, zeit) {
    var strom = {
        x: zeit,
        y: strom,
        type: 'scatter',
        name: 'Strom'
    };

    var spannung = {
        x: zeit,
        y: spannung,
        type: 'scatter',
        name: 'Spannung'
    };

    var data = [spannung, strom];

    var layout = {
        title: 'Strom und Spannung über die Zeit',
        xaxis: {
            title: 'Zeit in ms',
            showgrid: false,
            zeroline: false
        },
        yaxis: {
            title: 'Stärke Strom / Spannung in mA, mV',
            range: [0, 2],
            showline: false
        }
    };

    Plotly.newPlot(divID, data, layout);
}