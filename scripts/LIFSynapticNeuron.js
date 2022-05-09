// Ein LIF als synaptisches Neuron
// Es empfängt ein Spike als Input und behält es für eine bestimmte Zeit
class LIFSynapticNeuron extends LIFNeuron {
    constructor(n_syn, w, max_spikes = 50, u_rest = 0.0, u_thresh = 1.0, tau_rest = 4.0, r = 1.0, tau = 10.0, q = 1.5, tau_syn = 10.0) {
        super(u_rest, u_thresh, tau_rest, r, tau);

        // Number of synapses
        this.n_syn = n_syn;
        // The synaptic efficacy
        this.w = w;
        // Maximum number of spikes we remember
        this.max_spikes = max_spikes;
        // The neuron synaptic 'charge'
        this.q = q;
        // The synaptic time constant (ms)
        this.tau_syn = tau_syn;
    }

    // Update the parent graph variables and placeholders
    get_vars_and_ph() {
        // The history of synaptic spike times for the neuron
        // tf.Variable(tf.constant(-1.0, shape=[self.max_spikes, self.n_syn], dtype=tf.float32))
        //
        // https://www.w3docs.com/snippets/javascript/how-to-create-a-two-dimensional-array-in-javascript.html
        // https://developer.mozilla.org/de/docs/Web/JavaScript/Reference/Global_Objects/Array
        //
        // Ein Array Anzahl der Synapsen (Vertikal) mal Anzahl gemerkter Spikes (horizontal)
        // mit -1 vorbelegt
        //
        this.t_spikes = new Array(this.n_syn); // Anzahl Synapsen
        for (let i = 0; i < this.max_spikes; i++) {
            this.t_spikes[i] = new Array(this.max_spikes); // Anzahl der Spikes

            for (let ii = 0; ii < this.max_spikes; ii++) {
                this.t_spikes[i][ii] = -1;
            }
        }

        // The last index used to insert spike times
        // tf.Variable(self.max_spikes-1, dtype=tf.int32)
        this.t_spikes_idx = this.max_spikes - 1;

        // A placeholder indicating which synapse spiked in the last time step
        // tf.placeholder(shape=[self.n_syn], dtype=tf.bool)
        //
        // Ein Array Anzahl der Synapsen (Vertikal)
        // mit -1 vorbelegt
        //
        this.syn_has_spiked = new Array(this.n_syn);
    }

    // Operation to update spike times
    update_spike_times() {
        // Increase the age of older spikes
        // old_spikes_op = self.t_spikes.assign_add(tf.where(self.t_spikes >=0,
        //                                                   tf.constant(1.0, shape=[self.max_spikes, self.n_syn]) * self.dt,
        //                                                   tf.zeros([self.max_spikes, self.n_syn])))

        // Increment last spike index (modulo max_spikes)
        // new_idx_op = self.t_spikes_idx.assign(tf.mod(self.t_spikes_idx + 1, self.max_spikes))

        // Create a list of coordinates to insert the new spikes
        // idx_op = tf.constant(1, shape=[self.n_syn], dtype=tf.int32) * new_idx_op
        // coord_op = tf.stack([idx_op, tf.range(self.n_syn)], axis=1)

        // Create a vector of new spike times (non-spikes are assigned a negative time)
        // new_spikes_op = tf.where(self.syn_has_spiked,
        //                          tf.constant(0.0, shape=[self.n_syn]),
        //                          tf.constant(-1.0, shape=[self.n_syn]))

        // Replace older spikes by new ones
        // return tf.scatter_nd_update(old_spikes_op, coord_op, new_spikes_op)
    }

    // Override parent get_input_op method
    get_input_op() {
        // Update our memory of spike times with the new spikes
        let t_spikes_op = update_spike_times();

        //tf.where(t_spikes_op >=0,
        //   self.q/self.tau_syn * tf.exp(tf.negative(t_spikes_op/self.tau_syn)),
        //   t_spikes_op*0.0)
        // => Gibt ein 2D Shape zurück mit (row, col) der Elemente, die für die Condition True haben

        // Evaluate synaptic input current for each spike on each synapse
        let i_syn_op = new Array(this.n_syn); // Anzahl Synapsen
        for (let i = 0; i < this.n_syn; i++) {
            i_syn_op[i] = new Array(this.max_spikes); // Anzahl der Spikes

            for (let ii = 0; ii < this.max_spikes; ii++) {
                i_syn_op[i][ii] = 0;
            }
        }

        //= tf.where(t_spikes_op >=0,
        //                    self.q/self.tau_syn * tf.exp(tf.negative(t_spikes_op/self.tau_syn)),
        //                    t_spikes_op*0.0)
        //
        // Ich gehe durch t_spikes_op (die Dimension kenne ich) und prüfe den Eingangsstrom für
        // jeden gehaltenen Spike i_syn_op hält dann die aktuellen Werte
        for (let i = 0; i < this.n_syn; i++) {
            for (let ii = 0; ii < this.max_spikes; ii++) {
                let x = t_spikes_op[i][ii];

                if (x >= 0) {
                    // self.q/self.tau_syn * tf.exp(tf.negative(t_spikes_op/self.tau_syn))
                    // https://de.acervolima.com/tensorflow-js-tf-exp()-funktion/
                    // https://developer.mozilla.org/de/docs/Web/JavaScript/Reference/Global_Objects/Math/exp
                    let q_divby_tau = this.q / this.tau_syn; // self.q/self.tau_syn
                    let neg_t_spikes_op_divby_tau_syn = -1 * (x / this.tau_syn); // tf.exp(tf.negative(t_spikes_op/self.tau_syn))

                    i_syn_op[i][ii] = q_divby_tau * Math.exp(neg_t_spikes_op_divby_tau_syn);
                } else {
                    // t_spikes_op*0.0
                    i_syn_op[i][ii] = 0;
                }
            }
        }


        // Add each synaptic current to the input current
        //i_op =  tf.reduce_sum(self.w * i_syn_op)
        let i_op = 0;
        for (let i = 0; i < this.n_syn; i++) {
            for (let ii = 0; ii < this.max_spikes; ii++) {
                let x = t_spikes_op[i][ii];
                let i_op = x * this.w;
                this.i_app = this.i_app + i_op;
            }
        }
    }
}