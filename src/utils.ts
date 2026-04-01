import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

export interface MouthBox {
  left: number;
  top: number;
  size: number;
}

const MOUTH_BOX_MIN_SIDE_PX = 100;

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function getMouthBox(
  landmarks: NormalizedLandmark[], 
  width: number, 
  height: number
): MouthBox | null {
  // Mapping MediaPipe indices to MLKit logic
  const nose = landmarks[1];      // Nose Base
  const leftEye = landmarks[33];  // Left Eye
  const rightEye = landmarks[263]; // Right Eye

  if (!nose || !leftEye || !rightEye) return null;

  // Convert normalized (0-1) to pixel coordinates
  const nX = nose.x * width, nY = nose.y * height;
  const lX = leftEye.x * width, lY = leftEye.y * height;
  const rX = rightEye.x * width, rY = rightEye.y * height;

  const refLength = Math.hypot(rX - lX, rY - lY);
  const side = Math.max(MOUTH_BOX_MIN_SIDE_PX, refLength * 1.5);

  const maxLeft = Math.max(0, width - side);
  const maxTop = Math.max(0, height - side);

  return {
    left: Math.round(clamp(nX - refLength * 0.75, 0, maxLeft)),
    top: Math.round(clamp(nY - refLength * 0.2, 0, maxTop)),
    size: Math.round(side),
  };
}