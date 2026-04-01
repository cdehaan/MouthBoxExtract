import { JSDOM } from 'jsdom';

const dom = new JSDOM('', { url: "http://localhost" });
const { window } = dom;

// Safely polyfill only what's missing or needs overriding
const globals = {
  window: window,
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

// Specifically for MediaPipe's WASM loader which checks for 'self'
if (!('self' in global)) {
  (global as any).self = global;
}

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
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  return await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" },
    runningMode: "IMAGE",
    numFaces: 1
  });
}

async function processImages() {
  const detector = await setupDetector();
  const files = fs.readdirSync(INPUT_DIR).filter(f => /\.(jpe?g|png)$/i.test(f));

  console.log(`Found ${files.length} images. Starting extraction...`);

  for (const file of files) {
    const fileNameNoExt = path.parse(file).name;
    const outputFileName = `${fileNameNoExt}.png`;
    const outputPath = path.join(OUTPUT_DIR, outputFileName);

    // Skip if already processed
    if (fs.existsSync(outputPath)) continue;

    try {
      const inputPath = path.join(INPUT_DIR, file);
      const image = await loadImage(inputPath);
      
      // MediaPipe in Node needs a canvas-like source
      const canvas = createCanvas(image.width, image.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);

      const result = detector.detect(canvas as any);

      if (result.faceLandmarks && result.faceLandmarks.length > 0) {
        // Take the first face (most confident)
        const landmarks = result.faceLandmarks[0];
        const box = landmarks ? getMouthBox(landmarks, image.width, image.height) : null;

        if (box) {
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
          
          console.log(`✔ Processed: ${file}`);
        }
      } else {
        console.warn(`✘ No face detected: ${file}`);
      }
    } catch (err) {
      console.error(`Error processing ${file}:`, err);
    }
  }
  console.log('Done!');
}

processImages();