/* Edited from:
   @ https://examples.ml5js.org/javascript/imageclassification/imageclassification_doodlenet_canvas/
   @ https://examples.ml5js.org/javascript/sentiment/sentiment_interactive/
*/

// Initialize the Image Classifier method with DoodleNet.
let classifier;

let request;

// A variable to hold the canvas image we want to classify
let canvas, ctx;

// Two variable to hold the label and confidence of the result
let label;
let confidence;
let button;
const width = 280;
const height = 280;

let pX = null;
let pY = null;
let x = null;
let y = null;

let mouseDown = false;
let mobileMode = false;
let lastPt = null;
let islastPtNullPrev = false;

setup();
async function setup() {
  canvas = document.querySelector("#myCanvas");
  ctx = canvas.getContext("2d");

  classifier = await ml5.imageClassifier("DoodleNet", onModelReady);
  // Create a canvas with 280 x 280 px

  canvas.addEventListener("mousemove", onMouseUpdate);
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mouseup", onMouseUp);

  canvas.addEventListener("touchmove", onTouchUpdate);
  canvas.addEventListener("touchstart", onTouchStart);
  canvas.addEventListener("touchend", onTouchEnd);

  // Create a clear canvas button
  button = document.querySelector("#clearBtn");

  button.addEventListener("click", clearCanvas);
  // Create 'comment' div to hold results
  comment = document.querySelector("#comment");

  requestAnimationFrame(draw);


  response = document.querySelector("#response");
  // initialize sentiment
  sentiment = ml5.sentiment('movieReviews', onModelReady);

  inputBox = document.querySelector('#inputText');
  submitBtn = document.querySelector('#submitBtn');
  sentimentResult = document.querySelector('#score');

  // predicting the sentiment on mousePressed()
  submitBtn.addEventListener('click', getSentiment);

}

function getSentiment() {
  // get the values from the input
  const text = inputBox.value;
  // make the prediction
  const prediction = sentiment.predict(text);

  // display sentiment result on html page

  if (prediction.score < 0.4){
	response.textContent = `I'm sorry for the inconvenience, I hope it will be better soon :(`;
  } 
  else if (prediction.score > 0.6){
	response.textContent = `Thank you for your support, I really appreciate it :)`;
  }
  else{
  	response.textContent = `Really? Thank you anyway :|`;
  }
  
}


function onModelReady() {
  console.log("ready!");
}

function clearCanvas() {
  ctx.fillStyle = "#ebedef";
  ctx.fillRect(0, 0, width, height);
  comment.textContent = `I'm 100% sure it is the blank space`;
  lastPt = null;
  x = null;
  y = null;
}

function draw() {
  request = requestAnimationFrame(draw);

  if (mobileMode === false){
   if (pX == null || pY == null) {
    ctx.beginPath();
    ctx.fillStyle = "#ebedef";
    ctx.fillRect(0, 0, width, height);

    pX = x;
    pY = y;
   }

   // Set stroke weight to 10
   ctx.lineWidth = 10;
   // Set stroke color to black
   ctx.strokeStyle = "#000000";

   // If mouse is pressed, draw line between previous and current mouse positions
   if (mouseDown === true) {
     ctx.beginPath();
     ctx.lineCap = "round";
     ctx.moveTo(x, y);
     ctx.lineTo(pX, pY);
     ctx.stroke();
   }

   pX = x;
   pY = y;
  }
  else{
   //if (lastPt.x == null || lastPt.y == null) {
   // ctx.beginPath();
   // ctx.fillStyle = "#ebedef";
   // ctx.fillRect(0, 0, width, height);

   // lastPt.x = x;
   // lastPt.y = y;
   //}

   if(lastPt != null){
    
    if(islastPtNullPrev){
      // ctx.moveTo(x, y);
      islastPtNullPrev = false;
    }
    else{
      // Set stroke weight to 10
      ctx.lineWidth = 10;
      // Set stroke color to black
      ctx.strokeStyle = "#000000";
      ctx.beginPath();
      // ctx.lineCap = "round";
      ctx.moveTo(lastPt.x, lastPt.y);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
   }
   else{
    islastPtNullPrev = true;
   }

   if(x != null && y != null){
     lastPt = {x:x, y:y};
   }
  }

}

function onMouseDown(e) {
  mouseDown = true;
}

function onMouseUp(e) {
  mouseDown = false;
  classifyCanvas();
}

function onMouseUpdate(e) {
  const pos = getMousePos(canvas, e);
  x = pos.x;
  y = pos.y;
}

function onTouchStart(e){
  mobileMode = true;
  lastPt = null;
  x = null;
  y = null;
}

function onTouchEnd(e){
  e.preventDefault();
  
  // lastPt = null;
  // x = null;
  // y = null;
  mobileMode = false;
  classifyCanvas();
}

function onTouchUpdate(e){
  mobileMode = true;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  x = e.touches[0].pageX - rect.left;
  y = e.touches[0].pageY - rect.top;
}

function getMousePos(canvas, e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top,
  };
}

function classifyCanvas() {
  classifier.classify(canvas, gotResult);
}

// A function to run when we get any errors and the results
function gotResult(error, results) {
  // Display error in the console
  if (error) {
    console.error(error);
  }
  // The results are in an array ordered by confidence.
  console.log(results);

  comment.textContent = `I'm ${results[0].confidence.toFixed(4) * 100.0}% sure it is the ${results[0].label}!`;
}

