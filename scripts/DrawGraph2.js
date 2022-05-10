function DrawGraph2(divID, spannungIntern, SpannungExtern, strom, zeit) {
    var strom = {
        x: zeit,
        y: strom,
        type: 'scatter',
        name: 'Strom'
    };

    var spannungIntern = {
        x: zeit,
        y: spannungIntern,
        type: 'scatter',
        name: 'Spannung intern'
    };

    var spannungExtern = {
        x: zeit,
        y: SpannungExtern,
        type: 'scatter',
        name: 'Impuls'
    };

    var data = [spannungIntern, spannungExtern, strom];

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