// demo1();
// demo2();
// demo3();
// demo4();

function demo1() {
    // Ein skalar kann mit scalar engelegt werden. Mit print() erfolgt eine
    // Debugausgabe.
    let tensor = tf.scalar(1);
    tensor.print();

    // Es können Tensoren mit 0 erstellt werden (hier 1 dimensional)
    tensor = tf.zeros([2]);
    tensor.print();

    // Es können Tensoren mit 0 erstellt werden. Auf die Reihenfolge achten
    tensor = tf.zeros([3, 2]);
    tensor.print();

    // Tensor aus einem Array. Mehrere Arrays können verwendet werden. In diesem Fall
    // eine geschweifte Klammer um alle Elemente legen !
    tensor = tf.tensor([[11, 12], [21, 22], [31, 32]]);
    tensor.print();

    // Mit shape (Eigenschaft !) kann die Größe des Array ausgegeben werden. Hierbei auf die
    // Reihenfolge achten !
    console.log("return the shape ", tensor.shape);

    // Ein Tensor kann in ein Array zurück gewandelt werdenn. Hierbei kann auf flowing
    // API zurück gegriffen werden.
    tensor.array().then(x => console.log("return to array ", x));
}

function demo2() {
    // Mit dispose werden Speicherbelegungen frei gegeben.
    tf.dispose();

    let tensor = tf.tensor([[1, 2], [3, 4]]);
    tensor.print();

    // Mit square werden wird die einzelnen Werte quadriert
    tensor.square().print();

    // Mit add können Arrays aufaddiert werden. Hierbei wird chaining von
    // Operationen unterstützt
    tensor.square().add(tf.ones([2, 2])).print();

    // Mit ones kann analog zu zeros ein Tensor mit einsen angelegt werden. Hierbei
    // wird die size übergeben. Auf die Reihenfolge achten
    tensor = tf.ones([2, 3]);
    tensor.print();

    // Analog zu tf.dispose ist die Methode auch bei Variablen verfügbar
    tensor.dispose();
}

function demo3() {
    // Ein Tensor verfügt über eine dispose Methode. Alternativ kann auch tf.dispose
    // verwendet werden. Durch diese Methode werden nicht benötigte Tensoren im Speicher
    // frei gegeben. Mit tf.memory().numTensors kann die Anzahl der im Speicher
    // befindlichen Tensoren ausgegeben werden.
    const a = tf.tensor([1, 2, 3]);
    console.log(tf.memory().numTensors);
    a.dispose();
    console.log(tf.memory().numTensors);

    // tf.tidy ist ein besonderer Operator, der intern eine automatische Speicherbereinigung
    // bietet. Alle nicht meh benötigten Tensoren werden nach Verlassend er Methode
    // automatisch entsorgt, auch wenn nirgendwo dispose aufgerufen wurde.
    const b = tf.tidy(() => {
        let x = tf.tensor([1, 2, 3]);
        console.log(tf.memory().numTensors);
    })
    console.log(tf.memory().numTensors);
}

function demo4() {
    // Trainingsdaten (x) und Ergebnisdaten (y) festlegen. Hier ein XOR.
    const x = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]); // Dataset
    const y = tf.tensor2d([[0], [1], [1], [0]]); // labels

    // Ein sequential Objekt als Basis anlegen
    const model = tf.sequential();

    // Eingabelayer
    model.add(tf.layers.dense({
        units: 8,
        inputShape: 2,
        activation: 'relu'
    }))

    // Ausgabelayer
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))

    // Kompilieren des Modells
    model.compile({
        optimizer: 'sgd',
        loss: 'binaryCrossentropy'
    })

    // Trainieren des Modells
    model.fit(x, y, {
        batchSize: 1,
        epochs: 3000,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(epoch, logs.loss);
            }
        }
    })
}


// Simulation with square input currents
function pulseTest() {
    // Audiocontext, need a manual interaction within the Browser

    myAudioContext = new AudioContext();

    // Duration of the simulation in ms
    const T = 200;
    // Duration of each time step in ms
    const dt = 1;
    // Number of iterations = T/dt
    let steps = T / dt;
    // Output variables
    let timePoint = [];
    let I = [];
    let U = [];

    let neuron = new LIFNeuron();

    let neuron2 = new LIFNeuron();
    let I2 = [];
    let U2 = [];

    let neuron3 = new LIFNeuron();
    let I3 = [];
    let U3 = [];

    for (let x = 1; x <= steps; x++) {
        let i_app = 0;

        let t = x * dt;

        // Set input current in mA
        if (t > 10 && t < 30) {
            i_app = 0.5;
        } else if (t > 50 && t < 100) {
            i_app = 1.2;
        } else if (t > 120 && t < 180) {
            i_app = 1.5;
        } else {
            i_app = 0.0;
        }

        let newState = feedNeuron(neuron, i_app, dt);


        //neuron.i_app = i_app;
        //neuron.dt = 1;

        //neuron.get_potential_op();

        timePoint.push(t);
        //I.push(neuron.i_app);
        //U.push(neuron.u);

        I.push(newState.I);
        U.push(newState.U);

        let i_app2 = ((newState.U) / r1);
        let newState2 = feedNeuron(neuron2, i_app2, dt);

        I2.push(newState2.I);
        U2.push(newState2.U);

        let i_app3 = ((newState2.U) / 0.2);
        let newState3 = feedNeuron(neuron3, i_app3, dt);

        I3.push(newState3.I);
        U3.push(newState3.U);
    }

    DrawGraph("Diagramm1", U, I, timePoint);
    DrawGraph("Diagramm2", U2, I2, timePoint);
    DrawGraph("Diagramm3", U3, I3, timePoint);
}

let r1 = 0;

function dummy(value) {
    r1 = document.getElementById("range1").value;
    document.getElementById("curR1").innerText = "Aktueller Wert: " + r1.toString();

    pulseTest()
}

function feedNeuron(neuron, current, dt) {
    let out = {I: 0, U: 0, U_pulse: 0};

    neuron.i_app = current;
    neuron.dt = dt;

    neuron.get_potential_op();

    out.I = neuron.i_app;
    out.U = neuron.u;
    out.U_pulse = neuron.u_out;

    return out;
}

// Simulation with Random Current
function RandomTest() {
    // Audiocontext, need a manual interaction within the Browser
    myAudioContext = new AudioContext();
    // Duration of the simulation in ms
    const T = 200;
    // Duration of each time step in ms
    const dt = 1;
    // Number of iterations = T/dt
    let steps = T / dt;
    // Output variables
    let timePoint = [];
    let I = [];
    let U = [];

    let neuron = new LIFNeuron();
    const random = d3.randomNormal(1.5, 1.0);

    for (let x = 1; x <= steps; x++) {
        let i_app = 0;

        let t = x * dt;

        // Set input current in mA
        if (t > 10 && t < 180) {
            i_app = random();
        } else {
            i_app = 0.0;
        }

        neuron.i_app = i_app;
        neuron.dt = 1;

        neuron.get_potential_op();

        timePoint.push(t);
        I.push(neuron.i_app);
        U.push(neuron.u);
    }

    DrawGraph("Diagramm1", U, I, timePoint);
}


function dynamicImpulse(duration = 400, deltaTimesteps = 1, durationPulse = 30, directCurrent = 1.2) {
    let steps = duration / deltaTimesteps;

    let timePoint = [];
    let I = [];
    let U = [];
    let U_out = [];

    let pulseState = false;
    let timePulse = 0;

    let neuron = new LIFNeuron();

    for (let x = 1; x <= steps; x++) {
        let i_app = 0;

        let timeOverAll = x * deltaTimesteps;
        timePulse = timePulse + deltaTimesteps;

        if (timePulse >= durationPulse) {
            pulseState = !pulseState;
            timePulse = 0;
        }

        // Set input current in mA
        if (pulseState) {
            i_app = directCurrent;
        }

        let newState = feedNeuron(neuron, i_app, deltaTimesteps);

        timePoint.push(timeOverAll);
        I.push(newState.I);
        U.push(newState.U);
        U_out.push(newState.U_pulse);
    }

    DrawGraph2("Diagramm1", U, U_out, I, timePoint);
}