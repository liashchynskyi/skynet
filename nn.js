import {random, multiply, dotMultiply, mean, abs, subtract, transpose, add} from './mathjs'
import * as activation from './activations'
// Original work in JavaScript by Petro Liashchynskyi inspired by Andrew Trask with small improvements by Claude Coulombe
 
export class NeuralNetwork {
    constructor(...args) {
        this.input_nodes = args[0];
        this.hidden_nodes = args[1];
        this.output_nodes = args[2];

        this.epochs = 50000;
        this.activation = activation.sigmoid; 
        // this.lr = .5; // Better with this,lr=1.0
        this.lr = 1.0;
        this.output = 0; 

        //generate synapses
        // Better convergence with random weights between 0.5 to 1.0
        // Source: http://www.cs.stir.ac.uk/~kjt/techreps/pdf/TR148.pdf
        this.synapse0 = random([this.input_nodes, this.hidden_nodes], -1.0, 1.0);
        this.synapse1 = random([this.hidden_nodes, this.output_nodes], -1.0, 1.0);

    }
    setEpochs(numEpochs) {
        this.epochs = numEpochs;
    }
    setActivation(func) {
        switch (func) {
            case 'tanh': {
                this.activation = activation.tanh;
                break;
            }
            case 'relu': {
                this.activation = activation.relu;
                break;
            }
            case 'softplus': {
                this.activation = activation.softplus;
                break;
            }
            default: {
                this.activation = activation.sigmoid;
                break;
            } 
        }
    }
    setLearningRate(lr) {
        this.lr = lr;
    }
    train(input, target) {
        for (let i = 0; i < this.epochs; i++) {
            // Feed forward
            let input_layer = input;
            let hidden_layer_linear = multiply(input_layer, this.synapse0)
            let hidden_layer_activated = hidden_layer_linear.map(v => this.activation(v, false));
            let output_layer_linear = multiply(hidden_layer_activated, this.synapse1)
            let output_layer_activated = output_layer_linear.map(v => this.activation(v, false));
            // Compute output error
            let output_error = subtract(target, output_layer);
            // Backpropagation of the output error to the hidden layer 
            let output_delta = math.dotMultiply(output_error, math.multiply(hidden_layer_activated, this.weight1).map(v => this.activation(v, true)));            
            let hidden_error = multiply(output_delta, transpose(this.synapse1));
            // Backpropagation of the hidden layer error to the input layer
            let hidden_delta = dotMultiply(hidden_error, hidden_layer_linear.map(v => this.activation(v, true)));
            // Adjust weights of the output layer
            this.synapse1 = add(this.synapse1, multiply(transpose(hidden_layer_activated), multiply(output_delta, this.lr)));
            // Adjust weights of the hidden layer
            this.synapse0 = add(this.synapse0, multiply(transpose(input_layer), multiply(hidden_delta, this.lr)));
            this.output = output_layer;

            // Show error progression
            if (i % 10000 == 0)
                console.log(`Error: ${mean(abs(output_error))}`);
        }
    }
    predict(input) {
        let input_layer = input;
        let hidden_layer = multiply(input_layer, this.synapse0).map(v => this.activation(v, false));
        let output_layer = multiply(hidden_layer, this.synapse1).map(v => this.activation(v, false));
        return output_layer;
    }
}
