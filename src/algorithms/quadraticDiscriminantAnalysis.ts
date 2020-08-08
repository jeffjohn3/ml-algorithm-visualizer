import * as utils from "../utils";
var cov = require( 'compute-covariance' );

export class QDA {
    private means; 
    private covariances; 
    private gaus; 
    private covariance; 
    private offset; 
    private classes; 
    private prob; 

    constructor() {
    }

    train(training, labels) {
        this.means = []; 
        this.covariances = [];
        this.gaus = [];
        this.covariance = []; 
        this.offset = [];
        this.classes = [-1, 1];
        this.prob = [0, 0];

        this.classes.forEach((clas) => {
            let data = utils.where(training, labels, clas); 
            this.prob[clas] = data.length / training.length; 
            let mean = utils.mean0(data); 
            let covariance = cov(utils.transpose(data)); 
            
            this.means.push(mean); 
            this.covariances.push(covariance); 
            this.gaus.push(new utils.Gaussian(mean, covariance)); 
        });
    }
    classify(data) {
        let arr = [];
        for (let i = 0; i < data.length; i++) {
            let pdf0 = this.gaus[0].density(data); 
            let pdf1 = this.gaus[1].density(data);
            if (pdf0 > pdf1) {
                arr.push(-pdf0 / (pdf0 + pdf1)); 
            } 
            arr.push(pdf1 / (pdf0 + pdf1)); 
        }
        return arr; 
    }
}
