"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DecisionTree = exports.Logistic = exports.QDA = exports.LDA = void 0;
var utils_1 = require("./utils");
var math = require("mathjs");
var tf = require("@tensorflow/tfjs");
// import cov from 'compute-covariance'; 
var cov = require('compute-covariance');
var LDA = /** @class */ (function () {
    function LDA() {
    }
    LDA.prototype.train = function (training, labels) {
        var _this = this;
        this.means = [];
        this.covariances = [];
        this.gaus = [];
        this.covariance = [];
        this.offset = [];
        this.classes = [-1, 1];
        this.prob = [0, 0];
        this.classes.forEach(function (clas) {
            var data = utils_1.where(training, labels, clas);
            var covariance = cov(utils_1.transpose(data));
            _this.covariances.push(covariance);
            _this.covariance = utils_1.matSum(_this.covariance, covariance);
        });
        this.classes.forEach(function (clas) {
            var data = utils_1.where(training, labels, clas);
            _this.prob[clas] = data.length / training.length;
            var mean = utils_1.mean0(data);
            _this.means.push(mean);
            _this.gaus.push(new utils_1.Gaussian(mean, _this.covariance));
        });
    };
    LDA.prototype.classify = function (data) {
        var arr = [];
        for (var i = 0; i < data.length; i++) {
            var pdf0 = this.gaus[0].density(data);
            var pdf1 = this.gaus[1].density(data);
            if (pdf0 > pdf1) {
                arr.push(-pdf0 / (pdf0 + pdf1));
            }
            arr.push(pdf1 / (pdf0 + pdf1));
        }
        return arr;
    };
    return LDA;
}());
exports.LDA = LDA;
var QDA = /** @class */ (function () {
    function QDA() {
    }
    QDA.prototype.train = function (training, labels) {
        var _this = this;
        this.means = [];
        this.covariances = [];
        this.gaus = [];
        this.covariance = [];
        this.offset = [];
        this.classes = [-1, 1];
        this.prob = [0, 0];
        this.classes.forEach(function (clas) {
            var data = utils_1.where(training, labels, clas);
            _this.prob[clas] = data.length / training.length;
            var mean = utils_1.mean0(data);
            var covariance = cov(utils_1.transpose(data));
            _this.means.push(mean);
            _this.covariances.push(covariance);
            _this.gaus.push(new utils_1.Gaussian(mean, covariance));
        });
    };
    QDA.prototype.classify = function (data) {
        var arr = [];
        for (var i = 0; i < data.length; i++) {
            var pdf0 = this.gaus[0].density(data);
            var pdf1 = this.gaus[1].density(data);
            if (pdf0 > pdf1) {
                arr.push(-pdf0 / (pdf0 + pdf1));
            }
            arr.push(pdf1 / (pdf0 + pdf1));
        }
        return arr;
    };
    return QDA;
}());
exports.QDA = QDA;
var Logistic = /** @class */ (function () {
    function Logistic() {
    }
    Logistic.prototype.train = function (X, y, epsilon, lmbda, iterations) {
        if (epsilon === void 0) { epsilon = .01; }
        if (lmbda === void 0) { lmbda = .8; }
        if (iterations === void 0) { iterations = 1000; }
        var weights = math.zeros(X[0].length, 1);
        var grad;
        X = math.matrix(X);
        y = math.matrix(y);
        y.resize([y.size()[0], 1]);
        for (var i = 0; i < iterations; i++) {
            var C = math.subtract(y, utils_1.expit(math.multiply(X, weights).valueOf()));
            grad = utils_1.multcA(lmbda, math.multiply(utils_1.transpose(X.valueOf()), C).valueOf());
            weights = math.add(weights, utils_1.multcA(-epsilon, grad.valueOf()));
        }
        this.weights = weights;
    };
    Logistic.prototype.classify = function (X) {
        var prob = utils_1.expit([math.multiply(X, this.weights).valueOf()]);
        return [prob[0][0] * -2 + 1];
    };
    return Logistic;
}());
exports.Logistic = Logistic;
var EPS = .01;
var featureDivisions = 20;
var MAX_DEPTH = 7;
var DecisionTree = /** @class */ (function () {
    function DecisionTree(maxDepth) {
        if (maxDepth === void 0) { maxDepth = MAX_DEPTH; }
        this.maxDepth = maxDepth;
    }
    DecisionTree.prototype.informationGain = function (X, y, thresh) {
        var y1 = [], y2 = [];
        for (var i = 0; i < X.length; i++) {
            if (X[i] < thresh) {
                y1.push(y[i]);
            }
            else {
                y2.push(y[i]);
            }
        }
        return this.entropy(y) - (y1.length * this.entropy(y1) +
            y2.length * this.entropy(y2)) / y.length;
    };
    DecisionTree.prototype.entropy = function (y) {
        var total = 0;
        [-1, 1].forEach(function (label) {
            var count = 0;
            for (var i = 0; i < y.length; i++) {
                if (y[i] == label) {
                    count += 1;
                }
            }
            if (count != 0) {
                total += (count / y.length) * Math.log2(count / y.length);
            }
        });
        return -total;
    };
    DecisionTree.prototype.train = function (X, y) {
        this.X = X;
        this.y = y;
        var count = 0;
        this.y.forEach(function (element) {
            if (element == 1) {
                count += 1;
            }
        });
        this.pred = count / y.length;
        if (this.maxDepth > 0) {
            // 2D array of thresholds to split on. Shape: (#features x #featureDivisions) 
            var thresh = [];
            for (var i = 0; i < X[0].length; i++) {
                var col_1 = utils_1.getCol(X, i);
                thresh.push(tf.linspace(Math.min.apply(Math, col_1) + EPS, Math.max.apply(Math, col_1) - EPS, featureDivisions).arraySync());
            }
            // 2D of gains for each threshold.  Shape: (#features x #featureDivisions)
            var gains = [];
            for (var feature = 0; feature < X[0].length; feature++) {
                gains[feature] = [];
                for (var t = 0; t < thresh[feature].length; t++) {
                    gains[feature].push(this.informationGain(utils_1.getCol(X, feature), y, thresh[feature][t]));
                }
            }
            var m = utils_1.matArgMax(gains);
            this.feature = m[0];
            this.thresh = thresh[m[0]][m[1]];
            var col = utils_1.getCol(X, m[0]);
            var X0 = [], X1 = [], y0 = [], y1 = [];
            for (var i = 0; i < col.length; i++) {
                if (col[i] < this.thresh) {
                    X0.push(X[i]);
                    y0.push(y[i]);
                }
                else {
                    X1.push(X[i]);
                    y1.push(y[i]);
                }
            }
            if (X0.length > 0 && X1.length > 0) {
                this.left = new DecisionTree(this.maxDepth - 1);
                this.left.train(X0, y0);
                this.right = new DecisionTree(this.maxDepth - 1);
                this.right.train(X1, y1);
            }
            else {
                this.maxDepth = 0;
            }
        }
    };
    DecisionTree.prototype.classify = function (X) {
        if (this.maxDepth == 0) {
            return [this.pred * 2 - 1];
        }
        if (X[this.feature] < this.thresh) {
            return this.left.classify(X);
        }
        return this.right.classify(X);
    };
    DecisionTree.prototype.print = function () {
        console.log(this.maxDepth, this.pred, this.X.length, this.feature, this.thresh);
        if (this.maxDepth > 0) {
            this.left.print();
            this.right.print();
        }
    };
    return DecisionTree;
}());
exports.DecisionTree = DecisionTree;
//# sourceMappingURL=algorithms.js.map