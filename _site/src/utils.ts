import * as n from 'numeric'; 
import { maxPoolWithArgmax } from '@tensorflow/tfjs';

export function where(training, labels, label) {
    let arr = []; 
    for (let i = 0; i < labels.length; i++) {
        if (labels[i] == label) {
            arr.push(training[i]); 
        }
    }
    return arr; 
} 

// let arr = [[1,2], [3,4], [5,6]]; 
// let arr1 = [0, 1, 1]
// console.log(where(arr,  arr1, 0)); 

export function mean0(data) {
    let arr = []; 
    // console.log(data); 

    for (let i = 0; i < data[0].length; i++) {
        let sum = 0; 
        data.forEach(row => {
            sum += row[i];
        });
        arr.push(sum / data.length); 
    }
    return arr; 
}

export function transpose(data) {
    let arr = [];
    for (let i = 0; i < data[0].length; i++) {
        arr[i] = [];
        for (let j = 0; j < data.length; j++) {
            arr[i][j] = data[j][i];
        }
    }
    return arr; 
}

var sqrt2PI = Math.sqrt(Math.PI * 2);

export function Gaussian(m, s) {
    this.sigma = s;
    this.mu = m;
    this.k = this.mu.length; // dimension
    try {
        var det = n.det(this.sigma);
        this._sinv = n.inv(this.sigma); // π ^ (-1)
        this._coeff = 1 / (Math.pow(sqrt2PI, this.k) * Math.sqrt(det));
        if ( !(isFinite(det) && det > 0 && isFinite(this._sinv[0][0]))) {
            throw new Error("Invalid matrix");
        }
    } catch(e) {
        this._sinv = n.rep([this.k, this.k], 0);
        this._coeff = 0;
    }
}

Gaussian.prototype.density = function(x) {
    var delta = n.sub(x, this.mu); // 𝛿 = x - mu
    // Compute  Π = 𝛿T . Σ^(-1) . 𝛿
    var P = 0;
    for(var i=0; i<this.k; i++) {
        var sinv_line = this._sinv[i];
        var sum = 0;
        for(var j=0; j<this.k; j++) {
            sum += sinv_line[j] * delta[j];
        }
        P += delta[i] * sum
    }
    // Return: e^(-Π/2) / √|2.π.Σ|
    return this._coeff * Math.exp(P / -2);
};

export function matSum(A, B) {
    let C = []; 
    if (A.length < 1) {
        return B; 
    }
    if (B.length < 1) {
        return A; 
    }
    for (let i = 0; i < A.length; i++) {
        C[i] = []; 
        for (let j = 0; j < B.length; j++) {
            C[i][j] = A[i][j] + B[i][j]; 
        }
    }
    return C; 
}

export function zeros(x, y) {
    let A = []; 
    for (let i = 0; i < x; i++) {
        A[i] = []; 
        for (let j = 0; j < y; j++) {
            A[i][j] = 0;
        }
    }
    return A; 
}

export function expit(A) {
    let B = []; 
    for (let i = 0; i < A.length; i++) {
        B[i] = []; 
        for (let j = 0; j < A[i].length; j++) {
            B[i][j] = 1 / (1 + Math.exp(-A[i][j])); 
        }
    }
    return B; 
}

export function multcA(c, A) {
    let B = []; 
    for (let i = 0; i < A.length; i++) {
        B[i] = []; 
        for (let j = 0; j < A[i].length; j++) {
            B[i][j] = c * A[i][j]; 
        }
    }
    return B; 
}

export function getCol(matrix, col){
    var column = [];
    for(var i=0; i<matrix.length; i++){
       column.push(matrix[i][col]);
    }
    return column;
}

export function matArgMax(A) {
    let max = -999999; 
    let x, y; 
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < A[i].length; j++) {
            if (A[i][j] > max) {
                max = A[i][j]; 
                x = i; 
                y = j; 
            }
        }
    }
    return [x, y]; 
}
