import * as math from 'mathjs';
import * as utils from '../utils';

export class Logistic {
    private weights; 
    constructor() {
    }

    train(X, y, epsilon=.01, lmbda=.8, iterations=1000) {
        let weights = math.zeros(X[0].length, 1); 

        let grad; 
        X = math.matrix(X); 
        y = math.matrix(y); 
        y.resize([y.size()[0], 1]); 
        for (let i = 0; i < iterations; i++) {
            let C = math.subtract(y, utils.expit(math.multiply(X, weights).valueOf())); 
            grad = utils.multcA(lmbda, math.multiply(utils.transpose(X.valueOf()), C).valueOf());             
            weights = math.add(weights, utils.multcA(-epsilon, grad.valueOf())); 
            
        }
        this.weights = weights; 
    } 

    classify(X) {
        let prob = utils.expit([math.multiply(X, this.weights).valueOf()]); 
        return [prob[0][0] * -2 + 1]; 
    }

}
