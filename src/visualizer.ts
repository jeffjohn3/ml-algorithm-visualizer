import * as d3 from "d3"; 
import {Point, twoGaussians, circle, twoMoons} from "./data";
import {LDA, QDA, Logistic, DecisionTree, RandomTree} from './algorithms';
import * as fc from "d3fc"; 


// Parameters for dataset svg's.  
let margin = {top: 0, right: 0, bottom: 0, left: 0},
    width = 150 - margin.left - margin.right,
    height = 150 - margin.top - margin.bottom,
    radius = 3,
    color1 = 'MediumBlue',
    color2 = 'DarkOrange',
    numShades = 30,
    numSamples = 100,
    borderWidth = "1.1%", 
    padding = 0, 
    noise = 0;
const DENSITY = 100; 


//Training  
function train(model, points: Point[]) {
    let pointsArr = [];
    let labels = []
    points.forEach(function (point) {
        pointsArr.push([point.x, point.y]);
        labels.push(point.label); 
    }); 
    model.train(pointsArr, labels); 
}
// Generate heatmap 
function heatmap(model, gridx, gridy) {
    //Parameters for heatmap. 
    let xDomain = [-6, 6]; 

    // Canvas classification
    let xScale = d3.scaleLinear()
        .domain([0, DENSITY])
        .range(xDomain);
    let yScale = d3.scaleLinear()
        .domain([0, DENSITY])
        .range(xDomain);
    let canvasClass = [];
    for (let x = 0; x < DENSITY; x++) {
        canvasClass[x] = [] 
        for (let y = 0; y < DENSITY; y++) {
            canvasClass[x][y] = Math.pow(model.classify([xScale(x), yScale(DENSITY - y)])[0], 3); 
        }
    }
    // console.log(lda.classify([[1, 2]]));
    // console.log(canvasPoints); 

    // Heatmap. 
    let tmpScale = d3.scaleLinear<string, number>()
                        .domain([0, .5, 1])
                        .range([color1, "#e8eaeb", color2])
                        .clamp(true);
    let colors = d3.range(0, 1 + 1E-9, 1 / numShades).map(a => { return tmpScale(a); });
    let color = d3.scaleQuantize()
                        .domain([-1, 1])
                        .range(colors);
                        
                        
    let container = d3.select("#twoGaussians"); 
    let canvas = container.append("canvas")
                        .attr("width", numSamples)
                        .attr("height", numSamples)
                        .attr("class", "canvas")
                        .style("width", (width - 2 * padding) + "px")
                        .style("height", (height - 2 * padding) + "px")
                        .style("top", `${padding}px`)
                        .style("left", `${padding}px`)
                        .style("grid-column", gridy)
                        .style("grid-row", gridx);

    let context = (canvas.node() as HTMLCanvasElement).getContext("2d");
    let image = context.createImageData(DENSITY, DENSITY);   // dx and dy 
                    
    for (let y = 0, p = -1; y < DENSITY; ++y) {
        for (let x = 0; x < DENSITY; ++x) {
        let value = canvasClass[x][y];
        // console.log(d3.rgb(color(value))); 
        let c = d3.rgb(color(value));
        image.data[++p] = c.r;
        image.data[++p] = c.g;
        image.data[++p] = c.b;
        image.data[++p] = 160;
        }
    }
    context.putImageData(image, 0, 0);
}

function blankSvg(gridx, gridy) {
    let container = d3.select("#twoGaussians"); 
    let canvas = container.append("canvas")
                        .attr("width", numSamples)
                        .attr("height", numSamples)
                        .style("width", (width - 2 * padding) + "px")
                        .style("height", (height - 2 * padding) + "px")
                        .style("top", `${padding}px`)
                        .style("left", `${padding}px`)
                        .style("grid-column", gridy)
                        .style("grid-row", gridx);

    let context = (canvas.node() as HTMLCanvasElement).getContext("2d");
    let image = context.createImageData(DENSITY, DENSITY);   // dx and dy 
                    
    for (let y = 0, p = -1; y < DENSITY; ++y) {
        for (let x = 0; x < DENSITY; ++x) {
        // let value = canvasClass[x][y];
        // console.log(d3.rgb(color(value))); 
        // let c = d3.rgb(color(value));
        image.data[++p] = 192;
        image.data[++p] = 192;
        image.data[++p] = 192;
        image.data[++p] = 160;
        }
    }
    context.putImageData(image, 0, 0);
}

// Graph points on svg.  
function svg(points: Point[], gridx, gridy) {
    let x = d3.scaleLinear()
        .domain([-6, 6])
        .range([0, width]);
    let y = d3.scaleLinear()
        .domain([-6, 6])
        .range([height, 0]);
    let label = d3.scaleLinear<string>()
        .domain([-1, 1])
        .range([color1, color2]);
    let svg = d3.select("#twoGaussians")
        .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("class", "svg")
            .style("width", (width - 2 * padding) + "px")
            .style("height", (height - 2 * padding) + "px")
            .style("top", `${padding}px`)
            .style("left", `${padding}px`)
            .style("z-index", "4")
            .style("grid-column", gridy)
            .style("grid-row", gridx)
        .append("g")
            .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");
    svg.append("g")
        .selectAll("dot")
        .data(points)
        .enter()
        .append("circle")
            .attr("cx", function (d) { return x(d.x); })
            .attr("cy", function (d) { return y(d.y); })
            .attr("r", radius)
            .attr("stroke", "white") 
            .attr("stroke-width", borderWidth)
            .style("fill", function (d) { return label(d.label); })
            .style("z-index", "4");

            /* 
            svg.append("g")
                .attr("transform", "translate(0," + height + ")")
                .call(d3.axisBottom(x));
            svg.append("g")
                .call(d3.axisLeft(y)); 
            */
}


function everything(numSamples, noise) {
    let datasets: Point[][] = [twoGaussians(numSamples, noise), circle(numSamples, noise), twoMoons(numSamples, noise)]; 
    let models = [new LDA(), new QDA(), new Logistic(), new DecisionTree(), new RandomTree()];  

    for (let i = 0; i < datasets.length; i++) {

        let row = 2; 
        let col = 3; 
        svg(datasets[i], i + row, col); 
        blankSvg(i + row, col); 
        // j = 3 bc col1=filler and col2=justPoints
        for (let j = 0; j < models.length; j++) {
            let points: Point[] = datasets[i]; 
            let model = models[j];
            train(model, points); 
            heatmap(model, i + row, j + col + 1); 
            svg(points, i + row, j + col + 1);  
        }
    }
}

function clear() {
    function removeElements(className){
        var elements = document.getElementsByClassName(className);
        while(elements.length > 0){
            elements[0].parentNode.removeChild(elements[0]);
        }
    }
    removeElements("svg"); 
    removeElements("canvas"); 
}


var noiseSlider = (<HTMLInputElement> document.getElementById("noiseSlider")) ;
var noiseValue = document.getElementById("noiseValue");
noiseValue.innerHTML = noiseSlider.value;

noiseSlider.oninput = function() {
    noiseValue.innerHTML = this.value;
    clear(); 
    everything(parseInt(samplesSlider.value), parseInt(noiseSlider.value) / 100); 
}

var samplesSlider = (<HTMLInputElement> document.getElementById("samplesSlider")) ;
var samplesValue = document.getElementById("sampleValue");
samplesValue.innerHTML = samplesSlider.value;

samplesSlider.oninput = function() {
    samplesValue.innerHTML = this.value;
    clear(); 
    everything(parseInt(samplesSlider.value), parseInt(noiseSlider.value) / 100); 
}
everything(parseInt(samplesSlider.value), parseInt(noiseSlider.value) / 100); 

// let points: Point[] = datasets[0]; 
// let model = models[3];
// train(model, points); 
// console.log(model); 
// heatmap(model, 1, 1); 
// svg(points, 1, 1);  
