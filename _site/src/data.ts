import * as d3 from "d3"; 
import { POINT_CONVERSION_COMPRESSED } from "constants";
import { LayerVariable } from "@tensorflow/tfjs";

export type Point = {
    x: number, 
    y: number, 
    label: number, 
} 

function normalRandom(mean=0, variance=1): number {
    let a: number, b: number, s: number; 
    do {
        a = 2 * Math.random() - 1;
        b = 2 * Math.random() - 1;
        s = a * a + b * b;
    } while (s > 1);

    let result = Math.sqrt(-2 * Math.log(s) / s) * a; 
    return mean + Math.sqrt(variance) * result; 
} 

export function twoGaussians(numSamples: number, noise: number): Point[] { 
    let points: Point[] = []; 

    let varianceScale = d3.scaleLinear().domain([0, .5]).range([0.5, 4]);
    let variance = varianceScale(noise);
  
    function gaussian(muX: number, muY: number, label: number) {
        for (let i = 0; i < numSamples / 2; i++) {
            let x = normalRandom(muX, variance); 
            let y = normalRandom(muY, variance); 
            points.push({x, y, label}); 
        }
    }
    gaussian(2, 2, 1); 
    gaussian(-2, -2, -1); 


    return points; 
} 


/** Returns the eucledian distance between two points in space. */
function dist(a: Point, b: Point): number {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  
export function circle(numSamples: number, noise: number): Point[] {
  let points: Point[] = [];
  let radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = normalRandom(0, radius * 0.5);
    let angle = normalRandom(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = normalRandom(-radius, radius) * noise;
    let noiseY = normalRandom(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY, label: 0}, {x: 0, y: 0, label: 0});
    points.push({x, y, label: label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = normalRandom(radius * 0.7, radius);
    let angle = normalRandom(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = normalRandom(-radius, radius) * noise;
    let noiseY = normalRandom(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY, label: 0}, {x: 0, y: 0, label: 0});
    points.push({x, y, label: label});
  }
  return points;
}


export function twoMoons(numSamples: number, noise: number): Point[] {
    const K = 2.6;
    const A = .5;
    const parabola = ({ a, h, k, x }) => (a * Math.pow((x - h), 2)) + k;
    const getRandom = max => (Math.random() * max) - (max / 2);
    const getRandomPosition = (min, max) => min + (Math.random() * (max - min));
    const getDot = ({fn, noise, range, l}) => {
        const x = getRandomPosition(range[0], range[1]);
        const radius = getRandom(noise) * 10;
        const theta = getRandom(Math.PI * 2);

        return {x: x + (Math.sin(theta) * radius),
                y: fn(x) + (Math.cos(theta) * radius), 
                label: l};
    }

    let points: Point[] = []; 
    for (let i = 0; i < numSamples / 2; i++) {
        points.push(getDot({
            fn: x => parabola({ a: -1 * A, h: -4/3, k: K, x}),
            noise: noise,
            range: [-4, -4 + 16/3],
            l: 1
          })); 
    }    
    for (let i = 0; i < numSamples / 2; i++) {
        points.push(getDot({
            fn: x => parabola({ a: A, h: 4/3, k: -K, x}),
            noise: noise,
            range: [-4 + 8/3, 4],
            l: -1
          })); 
    }
    return points; 
}