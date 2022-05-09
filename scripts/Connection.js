class Connection {
    constructor(from, to, resistentMin, resistentMax) {
        this.from = from;
        this.to = to;
        this.resistentMin = resistentMin;
        this.resistentMay = resistentMax;

        // https://developer.mozilla.org/de/docs/Web/JavaScript/Reference/Global_Objects/Math/random
        this._resistant = Math.random() * (resistentMax - resistentMin) + min
    }
    //https://github.com/rafinskipg/neural-network-js/blob/master/src/connection.js
    get resistant() {
        return this._resistant;
    }

    set resistant(value) {
        this._resistant = value;
    }
}