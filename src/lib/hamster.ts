import type { PointCloudData } from './stereo';

/** Coordinates are normalised to the same central 4:3 crop used for stereo. */
export interface HamsterBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
  /** Model score, not a calibrated probability. */
  score: number;
}

export interface HamsterDistance {
  /** Approximate depth along the camera's optical axis, in metres. */
  metres: number | null;
  samples: number;
}

export interface HamsterObservation {
  frameId: number;
  capturedAt: number;
  leftSlot: 'a' | 'b';
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
  box: HamsterBox | null;
  distance: HamsterDistance | null;
}

const MODEL_SIZE = 300;
export const HAMSTER_SCORE_THRESHOLD = 0.25;

export interface ImageRegion { x: number; y: number; width: number; height: number }

/** Nine overlapping half-frame regions help with small animals. The centre is tried first. */
export function hamsterSearchRegions(width: number, height: number): ImageRegion[] {
  const w = Math.floor(width / 2);
  const h = Math.floor(height / 2);
  const xs = [Math.floor((width - w) / 2), 0, width - w];
  const ys = [Math.floor((height - h) / 2), 0, height - h];
  return ys.flatMap((y) => xs.map((x) => ({ x, y, width: w, height: h })));
}

export function extractRegion(rgba: Uint8ClampedArray, width: number, height: number, region: ImageRegion): Uint8ClampedArray {
  const { x, y, width: w, height: h } = region;
  if (rgba.length !== width * height * 4 || ![x, y, w, h].every(Number.isInteger) || x < 0 || y < 0 || w < 1 || h < 1 || x + w > width || y + h > height) {
    throw new Error('Invalid detector region');
  }
  const tile = new Uint8ClampedArray(w * h * 4);
  for (let row = 0; row < h; row++) {
    const start = ((y + row) * width + x) * 4;
    tile.set(rgba.subarray(start, start + w * 4), row * w * 4);
  }
  return tile;
}

/** Map a tile detection to the full crop; reject objects clipped by an internal tile edge. */
export function mapRegionBox(box: HamsterBox, region: ImageRegion, frameWidth: number, frameHeight: number): HamsterBox | null {
  const edge = 0.02;
  if ((region.x > 0 && box.left <= edge) || (region.y > 0 && box.top <= edge) ||
      (region.x + region.width < frameWidth && box.right >= 1 - edge) ||
      (region.y + region.height < frameHeight && box.bottom >= 1 - edge)) return null;
  return {
    left: (region.x + box.left * region.width) / frameWidth,
    right: (region.x + box.right * region.width) / frameWidth,
    top: (region.y + box.top * region.height) / frameHeight,
    bottom: (region.y + box.bottom * region.height) / frameHeight,
    score: box.score,
  };
}

/**
 * The original TF1 model uses ResizeBilinear with align_corners=false and
 * half_pixel_centers=false. Its input to this cut graph is RGB in [0, 255].
 */
export function prepareHamsterInput(rgba: Uint8ClampedArray, width: number, height: number): Float32Array {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || rgba.length !== width * height * 4) {
    throw new Error('Invalid detector frame');
  }
  const rgb = new Float32Array(MODEL_SIZE * MODEL_SIZE * 3);
  let out = 0;
  for (let y = 0; y < MODEL_SIZE; y++) {
    const sourceY = y * height / MODEL_SIZE;
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(height - 1, y0 + 1);
    const fy = sourceY - y0;
    for (let x = 0; x < MODEL_SIZE; x++) {
      const sourceX = x * width / MODEL_SIZE;
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(width - 1, x0 + 1);
      const fx = sourceX - x0;
      const a = (y0 * width + x0) * 4;
      const b = (y0 * width + x1) * 4;
      const c = (y1 * width + x0) * 4;
      const d = (y1 * width + x1) * 4;
      for (let channel = 0; channel < 3; channel++) {
        const upper = rgba[a + channel] + (rgba[b + channel] - rgba[a + channel]) * fx;
        const lower = rgba[c + channel] + (rgba[d + channel] - rgba[c + channel]) * fx;
        rgb[out++] = upper + (lower - upper) * fy;
      }
    }
  }
  return rgb;
}

/** Single highest-scoring valid hamster. Original model boxes are [ymin,xmin,ymax,xmax]. */
export function selectHamster(boxes: ArrayLike<number>, scores: ArrayLike<number>, threshold = HAMSTER_SCORE_THRESHOLD): HamsterBox | null {
  if (boxes.length !== scores.length * 4 || !scores.length || !Number.isFinite(threshold)) {
    throw new Error('Unexpected detector output');
  }
  const clamp = (value: number) => Math.min(1, Math.max(0, value));
  let best: HamsterBox | null = null;
  for (let i = 0; i < scores.length; i++) {
    const score = scores[i];
    const y0 = boxes[i * 4];
    const x0 = boxes[i * 4 + 1];
    const y1 = boxes[i * 4 + 2];
    const x1 = boxes[i * 4 + 3];
    if (![score, y0, x0, y1, x1].every(Number.isFinite) || score < threshold || score > 1 || (best && score <= best.score)) continue;
    const left = clamp(x0);
    const top = clamp(y0);
    const right = clamp(x1);
    const bottom = clamp(y1);
    if (right - left < 0.02 || bottom - top < 0.02) continue;
    best = { left, top, right, bottom, score };
  }
  return best;
}

/**
 * A box may contain background, so this is deliberately conservative. Only
 * matched points in its inner region count; require several spatially spread,
 * mutually consistent depths. Even then this is an approximation, not segmentation.
 */
export function estimateHamsterDepth(box: HamsterBox, cloud: PointCloudData): HamsterDistance {
  const spanX = box.right - box.left;
  const spanY = box.bottom - box.top;
  const left = (box.left + spanX * 0.2) * cloud.width;
  const right = (box.right - spanX * 0.2) * cloud.width;
  const top = (box.top + spanY * 0.2) * cloud.height;
  const bottom = (box.bottom - spanY * 0.2) * cloud.height;
  const midX = (left + right) / 2;
  const midY = (top + bottom) / 2;
  const depths: number[] = [];
  const quadrants = [0, 0, 0, 0];
  const count = Math.min(cloud.pixelIndices.length, Math.floor(cloud.positions.length / 3));
  for (let i = 0; i < count; i++) {
    const index = cloud.pixelIndices[i];
    if (index >= cloud.width * cloud.height) continue;
    const x = index % cloud.width;
    const y = Math.floor(index / cloud.width);
    const z = -cloud.positions[i * 3 + 2];
    if (x < left || x > right || y < top || y > bottom || !Number.isFinite(z) || z <= 0) continue;
    depths.push(z);
    quadrants[Number(x >= midX) + 2 * Number(y >= midY)]++;
  }
  if (depths.length < 8 || quadrants.filter((n) => n >= 2).length < 3) return { metres: null, samples: depths.length };
  depths.sort((a, b) => a - b);
  const median = depths[Math.floor(depths.length / 2)];
  const q1 = depths[Math.floor((depths.length - 1) * 0.25)];
  const q3 = depths[Math.floor((depths.length - 1) * 0.75)];
  if (q3 - q1 > median * 0.35) return { metres: null, samples: depths.length };
  return { metres: median, samples: depths.length };
}

export function describeHamsterPosition(box: HamsterBox): string {
  const x = (box.left + box.right) / 2;
  const y = (box.top + box.bottom) / 2;
  const horizontal = x < 1 / 3 ? 'слева' : x > 2 / 3 ? 'справа' : 'по центру';
  const vertical = y < 1 / 3 ? 'сверху' : y > 2 / 3 ? 'снизу' : 'посередине по высоте';
  return `${horizontal}, ${vertical}`;
}
