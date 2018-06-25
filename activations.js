import {exp, e, pow, log} from 'mathjs'

export function sigmoid(x, derivative) {
	let fx = 1 / (1 + exp(-x));
  if (derivative)
		return fx * (1 - fx);
	return fx;
}

export function tanh(x, derivative) {
	let fx = 2 / (1 + exp(-2 * x)) - 1;
	if (derivative) 
		return 1 - pow(fx, 2);
	return fx;
}

export function relu(x, derivative) {
	if (derivative)
		return x < 0 ? 0 : 1;
	return x < 0 ? 0 : x;
}

export function softplus(x, derivative) {
	if (derivative)
		return 1 / (1 + exp(-x));
	return log(1 + exp(x), e);
}