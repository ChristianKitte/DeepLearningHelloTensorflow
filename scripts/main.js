// demo1();
//demo2();
// demo3();
demo4();

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