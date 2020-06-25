"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.twoMoons = exports.circle = exports.twoGaussians = void 0;
var d3 = require("d3");
function normalRandom(mean, variance) {
    if (mean === void 0) { mean = 0; }
    if (variance === void 0) { variance = 1; }
    var a, b, s;
    do {
        a = 2 * Math.random() - 1;
        b = 2 * Math.random() - 1;
        s = a * a + b * b;
    } while (s > 1);
    var result = Math.sqrt(-2 * Math.log(s) / s) * a;
    return mean + Math.sqrt(variance) * result;
}
function twoGaussians(numSamples, noise) {
    var points = [];
    var varianceScale = d3.scaleLinear().domain([0, .5]).range([0.5, 4]);
    var variance = varianceScale(noise);
    function gaussian(muX, muY, label) {
        for (var i = 0; i < numSamples / 2; i++) {
            var x = normalRandom(muX, variance);
            var y = normalRandom(muY, variance);
            points.push({ x: x, y: y, label: label });
        }
    }
    gaussian(2, 2, 1);
    gaussian(-2, -2, -1);
    return points;
}
exports.twoGaussians = twoGaussians;
/** Returns the eucledian distance between two points in space. */
function dist(a, b) {
    var dx = a.x - b.x;
    var dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}
function circle(numSamples, noise) {
    var points = [];
    var radius = 5;
    function getCircleLabel(p, center) {
        return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
    }
    // Generate positive points inside the circle.
    for (var i = 0; i < numSamples / 2; i++) {
        var r = normalRandom(0, radius * 0.5);
        var angle = normalRandom(0, 2 * Math.PI);
        var x = r * Math.sin(angle);
        var y = r * Math.cos(angle);
        var noiseX = normalRandom(-radius, radius) * noise;
        var noiseY = normalRandom(-radius, radius) * noise;
        var label = getCircleLabel({ x: x + noiseX, y: y + noiseY, label: 0 }, { x: 0, y: 0, label: 0 });
        points.push({ x: x, y: y, label: label });
    }
    // Generate negative points outside the circle.
    for (var i = 0; i < numSamples / 2; i++) {
        var r = normalRandom(radius * 0.7, radius);
        var angle = normalRandom(0, 2 * Math.PI);
        var x = r * Math.sin(angle);
        var y = r * Math.cos(angle);
        var noiseX = normalRandom(-radius, radius) * noise;
        var noiseY = normalRandom(-radius, radius) * noise;
        var label = getCircleLabel({ x: x + noiseX, y: y + noiseY, label: 0 }, { x: 0, y: 0, label: 0 });
        points.push({ x: x, y: y, label: label });
    }
    return points;
}
exports.circle = circle;
function twoMoons(numSamples, noise) {
    var K = 2.6;
    var A = .5;
    var parabola = function (_a) {
        var a = _a.a, h = _a.h, k = _a.k, x = _a.x;
        return (a * Math.pow((x - h), 2)) + k;
    };
    var getRandom = function (max) { return (Math.random() * max) - (max / 2); };
    var getRandomPosition = function (min, max) { return min + (Math.random() * (max - min)); };
    var getDot = function (_a) {
        var fn = _a.fn, noise = _a.noise, range = _a.range, l = _a.l;
        var x = getRandomPosition(range[0], range[1]);
        var radius = getRandom(noise) * 10;
        var theta = getRandom(Math.PI * 2);
        return { x: x + (Math.sin(theta) * radius),
            y: fn(x) + (Math.cos(theta) * radius),
            label: l };
    };
    var points = [];
    for (var i = 0; i < numSamples / 2; i++) {
        points.push(getDot({
            fn: function (x) { return parabola({ a: -1 * A, h: -4 / 3, k: K, x: x }); },
            noise: noise,
            range: [-4, -4 + 16 / 3],
            l: 1
        }));
    }
    for (var i = 0; i < numSamples / 2; i++) {
        points.push(getDot({
            fn: function (x) { return parabola({ a: A, h: 4 / 3, k: -K, x: x }); },
            noise: noise,
            range: [-4 + 8 / 3, 4],
            l: -1
        }));
    }
    return points;
}
exports.twoMoons = twoMoons;
//# sourceMappingURL=data.js.map