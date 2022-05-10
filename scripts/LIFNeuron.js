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

        this.u_outValue = 1.0;
        this.u_out = 0;
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

        this.u_out = 0;
        return [this.u, this.t_rest, this.u_out];
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

        // signal
        beep(2, 400, 100);

        this.u_out = this.u_outValue;
        return [this.u, this.t_rest, this.u_out];
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

        this.u_out = 0;
        return [this.u, this.t_rest, this.u_out];
    }


    // Setzt den neuenStatus des Neuron
    get_potential_op() {
        if (this.t_rest > 0.0) {
            return this.get_resting_op();
        } else if (this.u > this.u_thresh) {
            this.get_firing_op();
        } else {
            this.get_integrating_op();
        }
    }
}
