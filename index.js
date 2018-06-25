import {NeuralNetwork} from './nn' 
import {matrix} from 'mathjs'

const input = matrix([[0,0], [0,1], [1,0], [1,1]]);
const target = matrix([[0], [1], [1], [0]]);


const nn = new NeuralNetwork(2, 4, 1);
nn.train(input, target);
console.log(nn.predict(input));
