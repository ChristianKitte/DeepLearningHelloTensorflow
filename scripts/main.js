// demo1();
//demo2();
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

// The browser will limit the number of concurrent audio contexts
// So be sure to re-use them whenever you can
let myAudioContext;

/**
 * Helper function to emit a beep sound in the browser using the Web Audio API.
 *
 * @param {number} duration - The duration of the beep sound in milliseconds.
 * @param {number} frequency - The frequency of the beep sound.
 * @param {number} volume - The volume of the beep sound.
 *
 * @returns {Promise} - A promise that resolves when the beep sound is finished.
 */
function beep(duration, frequency, volume) {
    return new Promise((resolve, reject) => {
        // Set default duration if not provided
        duration = duration || 200;
        frequency = frequency || 440;
        volume = volume || 100;

        try {
            let oscillatorNode = myAudioContext.createOscillator();
            let gainNode = myAudioContext.createGain();
            oscillatorNode.connect(gainNode);

            // Set the oscillator frequency in hertz
            oscillatorNode.frequency.value = frequency;

            // Set the type of oscillator
            oscillatorNode.type = "square";
            gainNode.connect(myAudioContext.destination);

            // Set the gain to the volume
            gainNode.gain.value = volume * 0.01;

            // Start audio with the desired duration
            oscillatorNode.start(myAudioContext.currentTime);
            oscillatorNode.stop(myAudioContext.currentTime + duration * 0.001);

            // Resolve the promise when the sound is finished
            oscillatorNode.onended = () => {
                resolve();
            };
        } catch (error) {
            reject(error);
        }
    });
}

// https://github.com/kaizouman/tensorsandbox/blob/master/snn/leaky_integrate_fire.ipynb
// https://js.tensorflow.org/api/latest/
class LIFNeuron {
    constructor(u_rest = 0.0, u_thresh = 1.0, tau_rest = 4.0, r = 1.0, tau = 10.0) {
        // Input current
        // Eingangsstrom
        this.i_app = 0.0;

        // The current membrane potential
        // Das aktuelle Membranpotenzial (Anfangswert auf u_rest)
        this.u = u_rest;
        // Membrane resting potential in mV
        // Ruhemembranpotential in mV
        this.u_rest = u_rest;
        // Membrane threshold potential in mV
        // Schwellenpotential der Membranen in mV
        this.u_thresh = u_thresh;

        // Membrane time constant in ms
        // Membranzeitkonstante in ms
        this.tau = tau;
        // Duration of the resting period in ms
        // Dauer der Ruhezeit in ms
        this.tau_rest = tau_rest;

        // Membrane resistance in Ohm
        // Membranwiderstand in Ohm
        this.r = r = 1.0;


        // The duration left in the resting period (0 most of the time except after a neuron spike)
        // Die in der Ruhephase verbleibende Dauer (0 für die meiste Zeit, außer nach einem Neuronenspike)
        this.t_rest = 0.0;
        // The chosen time interval for the stimulation in ms
        // Das gewählte Zeitintervall für die Stimulation in ms
        this.dt = 0.0;
    }

    // Neuron behaviour during integration phase (below threshold)
    // Verhalten der Neuronen während der Integrationsphase (unterhalb der Schwelle)
    get_integrating_op() {
        // Update membrane potential
        // Aktualisierung des Membranpotenzials
        let du_op = ((this.r * this.i_app) - this.u) / this.tau;
        this.u = this.u + (du_op * this.dt);

        // Refractory period is 0
        // Die Refraktärzeit beträgt 0
        this.t_rest = 0.0;

        return [this.u, this.t_rest];
    }

    // Neuron behaviour during firing phase (above threshold)
    // Verhalten der Neuronen während der Feuerungsphase (oberhalb der Schwelle)
    get_firing_op() {
        // Reset membrane potential
        // Membranpotenzial zurücksetzen
        this.u = this.u_rest;

        // Refractory period starts now
        // Die Refraktärzeit beginnt jetzt
        this.t_rest = this.tau_rest;

        return [this.u, this.t_rest];
    }

    // Neuron behaviour during resting phase (t_rest > 0)
    // Verhalten der Neuronen in der Ruhephase (t_rest > 0)
    get_resting_op() {
        // Membrane potential stays at u_rest
        // Das Membranpotenzial bleibt bei u_rest
        this.u = this.u_rest;

        // Refractory period is decreased by dt
        // Die Refraktärzeit verringert sich um dt
        this.t_rest = this.t_rest - this.dt;

        return [this.u, this.t_rest];
    }


    // Setzt den neuenStatus des Neuron
    get_potential_op() {
        if (this.t_rest > 0.0) {
            return this.get_resting_op();
        } else if (this.u > this.u_thresh) {
            this.get_firing_op();
            beep(2, 400, 100);
            console.log("beep");
        } else {
            this.get_integrating_op();
        }
    }
}

// Simulation with square input currents
function test() {
    myAudioContext = new AudioContext();
    // Duration of the simulation in ms
    const T = 200;
    // Duration of each time step in ms
    const dt = 1;
    // Number of iterations = T/dt
    let steps = T / dt;
    // Output variables
    let I = [];
    let U = [];

    let neuron = new LIFNeuron();

    for (x = 1; x <= steps; x++) {
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

        neuron.i_app = i_app;
        neuron.dt = 1;

        neuron.get_potential_op();
        //feed = {i_app: i_app, dt: dt};

        //u = sess.run(neuron.potential, feed_dict = feed)

        console.log(neuron.u);
        I.push(neuron.i_app);
        U.push(neuron.u);
    }

    //console.log(I.toString());
    //console.log(U.toString());
    //console.log(U.length);
}

//test();
