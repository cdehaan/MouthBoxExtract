import { JSDOM } from 'jsdom';

// Mock the browser environment
const dom = new JSDOM('', { url: "http://localhost" });
const { window } = dom;
const globals = {
  window,
  document: window.document,
  Node: window.Node,
  HTMLCanvasElement: window.HTMLCanvasElement,
  HTMLImageElement: window.HTMLImageElement,
  ImageData: (window as any).ImageData,
};

Object.entries(globals).forEach(([key, value]) => {
  if (!(key in global)) {
    Object.defineProperty(global, key, { value, writable: true, configurable: true });
  }
});

if (!('self' in global)) { (global as any).self = global; }

import fs from 'fs';
import path from 'path';
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';
import { createCanvas, loadImage } from 'canvas';
import sharp from 'sharp';
import { getMouthBox } from './utils.js';

const INPUT_DIR = './input';
const OUTPUT_DIR = './output';
const MODEL_PATH = './models/face_landmarker.task';

async function setupDetector() {
  console.log("--- Initializing MediaPipe ---");
  try {
    // Use a local absolute path to avoid fetch/network hangs
    const wasmPath = path.resolve("./node_modules/@mediapipe/tasks-vision/wasm");
    
    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    
    console.log("--- WASM Loaded. Loading Model ---");
    
    const landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { 
        modelAssetPath: path.resolve(MODEL_PATH),
        delegate: "CPU" 
      },
      runningMode: "IMAGE",
      numFaces: 1
    });
    console.log("--- Detector Ready ---");
    return landmarker;
  } catch (err) {
    console.error("Failed to initialize MediaPipe:", err);
    throw err;
  }
}

async function processImages() {
  const absoluteInputPath = path.resolve(INPUT_DIR);
  console.log(`Checking for images in: ${absoluteInputPath}`);

  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR);
  }

  const detector = await setupDetector();
  const files = fs.readdirSync(INPUT_DIR).filter(f => /\.(jpe?g|png)$/i.test(f));

  console.log(`--- Scan Complete: Found ${files.length} images in ${INPUT_DIR} ---`);

  for (const file of files) {
    const fileNameNoExt = path.parse(file).name;
    const outputPath = path.join(OUTPUT_DIR, `${fileNameNoExt}.png`);

    if (fs.existsSync(outputPath)) {
      console.log(`[Skip] ${file} already exists in output.`);
      continue;
    }

    console.log(`[Processing] ${file}...`);

    try {
      const inputPath = path.join(INPUT_DIR, file);
      const image = await loadImage(inputPath);
      console.log(`  -> Image loaded: ${image.width}x${image.height}`);
      
      const canvas = createCanvas(image.width, image.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);

      // Perform detection
      const result = detector.detect(canvas as any);

      if (result.faceLandmarks && result.faceLandmarks.length > 0) {
        console.log(`  -> Face detected! Finding mouth box...`);
        const landmarks = result.faceLandmarks[0];
        const box = landmarks ? getMouthBox(landmarks, image.width, image.height) : null;

        if (box) {
          console.log(`  -> Mouth Box: x:${box.left}, y:${box.top}, size:${box.size}`);
          
          await sharp(inputPath)
            .extract({ 
                left: box.left, 
                top: box.top, 
                width: box.size, 
                height: box.size 
            })
            .resize(320, 320)
            .png()
            .toFile(outputPath);
          
          console.log(`  -> SUCCESS: Saved to ${outputPath}`);
        } else {
          console.warn(`  -> [Fail] Landmarks found but box calculation failed.`);
        }
      } else {
        console.warn(`  -> [Fail] No face detected by MediaPipe.`);
      }
    } catch (err) {
      console.error(`  -> [Error] Failed to process ${file}:`, err);
    }
  }
  console.log("--- All Done ---");
}

processImages().then(() => {
  console.log("--- Process Execution Finished ---");
}).catch(err => console.error("Critical Error:", err));