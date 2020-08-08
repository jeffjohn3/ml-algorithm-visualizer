import * as utils from '../utils'; 
import * as tf from "@tensorflow/tfjs";

let EPS = .01; 
let featureDivisions = 20; 
let MAX_DEPTH = 7; 

export class DecisionTree {
    private maxDepth; 
    private feature; 
    private thresh; 
    private left; 
    private right; 
    private X; 
    private y; 
    private pred; 

    constructor(maxDepth=MAX_DEPTH) {
        this.maxDepth = maxDepth; 
    }

    informationGain(X, y, thresh) {
        let y1 = [], y2 = []; 
        for (let i = 0; i < X.length; i++) {
            if (X[i] < thresh) {
                y1.push(y[i]);
            }
            else {
                y2.push(y[i]);
            }
        }
        return this.entropy(y) - (y1.length * this.entropy(y1) + 
                                  y2.length * this.entropy(y2))  / y.length; 
    }

    entropy(y) {
        let total = 0; 
        [-1, 1].forEach((label) => {
            let count = 0; 
            for (let i = 0; i < y.length; i++) {
                if (y[i] == label) {
                    count += 1; 
                }
            }
            if (count != 0) {
                total += (count / y.length) * Math.log2(count / y.length); 
            }
        }); 
        return -total; 
    }

    train(X, y) {
        this.X = X; 
        this.y = y; 
        let count = 0; 
        this.y.forEach(element => {
            if (element == 1) {
                count += 1; 
            }
        });
        this.pred = count / y.length; 
        if (this.maxDepth > 0) {
            // 2D array of thresholds to split on. Shape: (#features x #featureDivisions) 
            let thresh = [];
            for (let i = 0; i < X[0].length; i++) {
                let col = utils.getCol(X, i); 
                thresh.push(tf.linspace(Math.min(...col) + EPS, Math.max(...col) - EPS, featureDivisions).arraySync()); 
            }
            // 2D of gains for each threshold.  Shape: (#features x #featureDivisions)
            let gains = []; 
            for (let feature = 0; feature < X[0].length; feature++) {
                gains[feature] = []; 
                for (let t = 0; t < thresh[feature].length; t++) {
                    gains[feature].push(this.informationGain(utils.getCol(X, feature), y, thresh[feature][t])); 
                }
            }
            let m = utils.matArgMax(gains); 
            this.feature = m[0]; 
            this.thresh = thresh[m[0]][m[1]]; 
            let col = utils.getCol(X, m[0]); 
            let X0 = [], X1 = [], y0 = [], y1 = []; 
            for (let i = 0; i < col.length; i++) {
                if (col[i] < this.thresh) {
                    X0.push(X[i]);
                    y0.push(y[i]); 
                } else {
                    X1.push(X[i]); 
                    y1.push(y[i]); 
                }
            }
            if (X0.length > 0 && X1.length > 0) {
                this.left = new DecisionTree(this.maxDepth - 1); 
                this.left.train(X0, y0); 
                this.right = new DecisionTree(this.maxDepth - 1); 
                this.right.train(X1, y1); 
            } else {
                this.maxDepth = 0; 
            }
        }
    } 
    
    classify(X) {
        if (this.maxDepth == 0) {
            return [this.pred * 2 - 1]; 
        }
        if (X[this.feature] < this.thresh) {
            return this.left.classify(X); 
        } 
        return this.right.classify(X); 
    }

    print() {
        console.log(this.maxDepth, this.pred, this.X.length, this.feature, this.thresh); 
        if (this.maxDepth > 0) {
            this.left.print(); 
            this.right.print(); 
        }
    }
}


export class RandomTree {
    private maxDepth; 
    private feature; 
    private thresh; 
    private left; 
    private right; 
    private X; 
    private y; 
    private pred; 

    constructor(maxDepth=MAX_DEPTH) {
        this.maxDepth = maxDepth; 
    }

    informationGain(X, y, thresh) {
        let y1 = [], y2 = []; 
        for (let i = 0; i < X.length; i++) {
            if (X[i] < thresh) {
                y1.push(y[i]);
            }
            else {
                y2.push(y[i]);
            }
        }
        return this.entropy(y) - (y1.length * this.entropy(y1) + 
                                  y2.length * this.entropy(y2))  / y.length; 
    }

    entropy(y) {
        let total = 0; 
        [-1, 1].forEach((label) => {
            let count = 0; 
            for (let i = 0; i < y.length; i++) {
                if (y[i] == label) {
                    count += 1; 
                }
            }
            if (count != 0) {
                total += (count / y.length) * Math.log2(count / y.length); 
            }
        }); 
        return -total; 
    }

    train(X, y) {
        this.X = X; 
        this.y = y; 
        let count = 0; 
        this.y.forEach(element => {
            if (element == 1) {
                count += 1; 
            }
        });
        this.pred = count / y.length; 
        if (this.maxDepth > 0) {
            // 2D array of thresholds to split on. Shape: (#features x #featureDivisions) 
            let thresh = [];  
            for (let i = 0; i < X[0].length; i++) {
                let col = utils.getCol(X, i); 
                thresh.push(tf.linspace(Math.min(...col) + EPS, Math.max(...col) - EPS, featureDivisions).arraySync()); 
            }
            // 2D of gains for each threshold.  Shape: (#features x #featureDivisions)
            let gains = []; 
            let f = (Math.random() > .5) ? 1 : 0; 
            for (let feature = 0; feature < X[0].length; feature++) {
                gains[feature] = []; 
                for (let t = 0; t < thresh[feature].length; t++) {
                    if (f == feature) {
                        gains[feature].push(this.informationGain(utils.getCol(X, feature), y, thresh[feature][t])); 
                    } else {
                        gains[feature].push(0); 
                    }
                }
            }
            let m = utils.matArgMax(gains); 
            this.feature = m[0]; 
            this.thresh = thresh[m[0]][m[1]]; 
            let col = utils.getCol(X, m[0]); 
            let X0 = [], X1 = [], y0 = [], y1 = []; 
            for (let i = 0; i < col.length; i++) {
                if (col[i] < this.thresh) {
                    X0.push(X[i]);
                    y0.push(y[i]); 
                } else {
                    X1.push(X[i]); 
                    y1.push(y[i]); 
                }
            }
            if (X0.length > 0 && X1.length > 0) {
                this.left = new DecisionTree(this.maxDepth - 1); 
                this.left.train(X0, y0); 
                this.right = new DecisionTree(this.maxDepth - 1); 
                this.right.train(X1, y1); 
            } else {
                this.maxDepth = 0; 
            }
        }
    } 
    
    classify(X) {
        if (this.maxDepth == 0) {
            return [this.pred * 2 - 1]; 
        }
        if (X[this.feature] < this.thresh) {
            return this.left.classify(X); 
        } 
        return this.right.classify(X); 
    }

    print() {
        console.log(this.maxDepth, this.pred, this.X.length, this.feature, this.thresh); 
        if (this.maxDepth > 0) {
            this.left.print(); 
            this.right.print(); 
        }
    }
}