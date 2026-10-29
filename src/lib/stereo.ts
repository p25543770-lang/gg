/**
 * A small, deliberately conservative stereo matcher for approximately parallel,
 * unmirrored cameras. It does not replace a calibrated stereo-rectification rig.
 * All coordinates and disparities below refer to the same cropped input image.
 */
export interface Frame {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export type CameraOrder = 'a-left' | 'b-left';

export interface OrientationResult {
  order: CameraOrder | null;
  matches: number;
  agreement: number;
  /** Median y(A) - y(B), in pixels of the processed frame. */
  verticalOffset: number;
  medianDisparity: number;
  /** Fraction of vertical strips of frame A covered by inlier matches, 0..1. */
  overlapRatio: number;
  /** Interquartile spread of inlier disparities, pixels. */
  disparitySpread: number;
  /** Mean local texture energy of each frame (higher = more detail). */
  sharpnessA: number;
  sharpnessB: number;
  /** meanLuminance(A) / meanLuminance(B); 1 = equal exposure. */
  brightnessRatio: number;
}

export interface StereoSettings {
  baselineCm: number;
  horizontalFov: number;
  maxDepthM: number;
  /** y(left) - y(right), in processed pixels. */
  verticalOffset: number;
  sampleStep?: number;
}

export interface PointCloudData {
  positions: Float32Array;
  rgbColors: Float32Array;
  depthColors: Float32Array;
  depthMap: Uint8ClampedArray;
  /** Index in the left 4:3 frame for each corresponding position. */
  pixelIndices: Uint32Array;
  width: number;
  height: number;
  count: number;
  near: number | null;
  far: number | null;
}

export interface Descriptors {
  width: number;
  height: number;
  gray: Uint8Array;
  low: Uint32Array;
  high: Uint32Array;
}

export interface Match {
  x: number;
  y: number;
  cost: number;
}

const RADIUS = 3;
const MAX_COST = 16;
const MIN_MARGIN = 2;

function validateFrame(frame: Frame) {
  if (
    !Number.isInteger(frame.width) ||
    !Number.isInteger(frame.height) ||
    frame.width < 16 ||
    frame.height < 16 ||
    frame.data.length !== frame.width * frame.height * 4
  ) {
    throw new Error('Invalid image frame');
  }
}

export function describe(frame: Frame): Descriptors {
  validateFrame(frame);
  const { width, height, data } = frame;
  const gray = new Uint8Array(width * height);
  const low = new Uint32Array(width * height);
  const high = new Uint32Array(width * height);

  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = (77 * data[p] + 150 * data[p + 1] + 29 * data[p + 2]) >> 8;
  }

  // 7×7 census transform. Comparing local brightness ranks rather than raw
  // RGB makes the correspondence less sensitive to different exposures.
  for (let y = RADIUS; y < height - RADIUS; y++) {
    for (let x = RADIUS; x < width - RADIUS; x++) {
      const index = y * width + x;
      const center = gray[index];
      let a = 0;
      let b = 0;
      let bit = 0;
      for (let dy = -RADIUS; dy <= RADIUS; dy++) {
        for (let dx = -RADIUS; dx <= RADIUS; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (gray[index + dy * width + dx] < center) {
            if (bit < 24) a |= 1 << bit;
            else b |= 1 << (bit - 24);
          }
          bit++;
        }
      }
      low[index] = a;
      high[index] = b;
    }
  }
  return { width, height, gray, low, high };
}

function popcount(value: number): number {
  value -= (value >>> 1) & 0x55555555;
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function distance(a: Descriptors, ai: number, b: Descriptors, bi: number): number {
  return popcount(a.low[ai] ^ b.low[bi]) + popcount(a.high[ai] ^ b.high[bi]);
}

export function texture(image: Descriptors, x: number, y: number): number {
  const { gray, width: w } = image;
  const i = y * w + x;
  const gx =
    Math.abs(gray[i - 1] - gray[i + 1]) +
    Math.abs(gray[i - w - 1] - gray[i - w + 1]) +
    Math.abs(gray[i + w - 1] - gray[i + w + 1]);
  const gy =
    Math.abs(gray[i - w] - gray[i + w]) +
    Math.abs(gray[i - w - 1] - gray[i + w - 1]) +
    Math.abs(gray[i - w + 1] - gray[i + w + 1]);
  return Math.min(gx, gy);
}

export function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function featurePoints(image: Descriptors): Array<{ x: number; y: number }> {
  const points: Array<{ x: number; y: number }> = [];
  const tile = 16;
  for (let top = RADIUS + 1; top < image.height - RADIUS - 1; top += tile) {
    for (let left = RADIUS + 1; left < image.width - RADIUS - 1; left += tile) {
      let best = 32;
      let point: { x: number; y: number } | null = null;
      const bottom = Math.min(top + tile, image.height - RADIUS - 1);
      const right = Math.min(left + tile, image.width - RADIUS - 1);
      for (let y = top; y < bottom; y += 2) {
        for (let x = left; x < right; x += 2) {
          const score = texture(image, x, y);
          if (score > best) {
            best = score;
            point = { x, y };
          }
        }
      }
      if (point) points.push(point);
    }
  }
  return points;
}

/** Search both directions for a distinctive feature during auto ordering. */
function searchArea(
  source: Descriptors,
  target: Descriptors,
  x: number,
  y: number,
  maxDx: number,
  maxDy: number,
): Match | null {
  const { width: w, height: h } = source;
  const minX = Math.max(RADIUS, x - maxDx);
  const maxX = Math.min(w - RADIUS - 1, x + maxDx);
  const minY = Math.max(RADIUS, y - maxDy);
  const maxY = Math.min(h - RADIUS - 1, y + maxDy);
  const searchWidth = maxX - minX + 1;
  const costs = new Uint8Array(searchWidth * (maxY - minY + 1));
  const sourceIndex = y * w + x;
  let bestCost = 255;
  let bestX = x;
  let bestY = y;
  let i = 0;
  for (let ty = minY; ty <= maxY; ty++) {
    for (let tx = minX; tx <= maxX; tx++) {
      const cost = distance(source, sourceIndex, target, ty * w + tx);
      costs[i++] = cost;
      if (cost < bestCost) {
        bestCost = cost;
        bestX = tx;
        bestY = ty;
      }
    }
  }
  if (bestCost > MAX_COST) return null;
  let second = 255;
  i = 0;
  for (let ty = minY; ty <= maxY; ty++) {
    for (let tx = minX; tx <= maxX; tx++) {
      const cost = costs[i++];
      // Neighbouring pixels of the same feature aren't a competing match.
      if (Math.abs(tx - bestX) <= 2 && Math.abs(ty - bestY) <= 2) continue;
      if (cost < second) second = cost;
    }
  }
  if (second - bestCost < MIN_MARGIN) return null;
  return { x: bestX, y: bestY, cost: bestCost };
}

/**
 * Infer camera order from the sign of mutually checked horizontal disparities.
 * A positive x(A)-x(B) means A is left for parallel cameras. No guess is
 * returned if the sign is ambiguous or there aren't enough shared features.
 * The same pass reports real pair metrics (overlap, spread, sharpness, exposure)
 * used by auto setup — every number is measured, never invented.
 */
export function inferCameraOrder(frameA: Frame, frameB: Frame): OrientationResult {
  if (frameA.width !== frameB.width || frameA.height !== frameB.height) {
    throw new Error('Stereo frames must have the same size');
  }
  const a = describe(frameA);
  const b = describe(frameB);
  const maxDx = Math.max(4, Math.floor(a.width * 0.32));
  const maxDy = Math.max(3, Math.floor(a.height * 0.055));
  const matches: Array<{ x: number; dx: number; dy: number; bx: number }> = [];
  const featureList = featurePoints(a);
  let sharpA = 0;
  let sharpB = 0;

  for (const point of featureList) {
    sharpA += texture(a, point.x, point.y);
    const forward = searchArea(a, b, point.x, point.y, maxDx, maxDy);
    if (!forward || Math.abs(point.x - forward.x) < 2) continue;
    sharpB += texture(b, forward.x, forward.y);
    const reverse = searchArea(b, a, forward.x, forward.y, maxDx, maxDy);
    if (!reverse || Math.abs(reverse.x - point.x) > 1 || Math.abs(reverse.y - point.y) > 1) continue;
    matches.push({ x: point.x, dx: point.x - forward.x, dy: point.y - forward.y, bx: forward.x });
  }

  const dy = median(matches.map((match) => match.dy));
  // A consistent vertical offset is a useful sanity check, not a substitute
  // for lens calibration or full stereo rectification.
  const inliers = matches.filter((match) => Math.abs(match.dy - dy) <= 3);
  const positive = inliers.filter((match) => match.dx > 0).length;
  const negative = inliers.length - positive;
  const agreement = inliers.length ? Math.max(positive, negative) / inliers.length : 0;
  const signedMedian = median(inliers.map((match) => match.dx));
  const order =
    inliers.length >= 8 && agreement >= 0.72 && Math.abs(signedMedian) >= 2
      ? signedMedian > 0
        ? 'a-left'
        : 'b-left'
      : null;

  // Overlap: how much of the left width actually found something in the right.
  const strips = 16;
  const covered = new Array<boolean>(strips).fill(false);
  for (const match of inliers) covered[Math.min(strips - 1, Math.floor((match.x / a.width) * strips))] = true;
  const overlapRatio = covered.filter(Boolean).length / strips;
  const sortedDx = inliers.map((match) => match.dx).sort((x, y) => x - y);
  const disparitySpread = sortedDx.length >= 4
    ? sortedDx[Math.floor((sortedDx.length - 1) * 0.75)] - sortedDx[Math.floor((sortedDx.length - 1) * 0.25)]
    : 0;

  let sumA = 0;
  let sumB = 0;
  let countA = 0;
  for (let i = 0; i < a.gray.length; i += 7) {
    sumA += a.gray[i];
    sumB += b.gray[i];
    countA++;
  }
  const brightnessRatio = countA && sumB > 0 ? sumA / sumB : 1;

  return {
    order,
    matches: inliers.length,
    agreement,
    verticalOffset: Math.round(dy),
    medianDisparity: signedMedian,
    overlapRatio,
    disparitySpread,
    sharpnessA: featureList.length ? sharpA / featureList.length : 0,
    sharpnessB: featureList.length && matches.length ? sharpB / matches.length : 0,
    brightnessRatio,
  };
}

/** Search an approximately rectified epipolar line in the specified direction. */
export function searchLine(
  source: Descriptors,
  target: Descriptors,
  x: number,
  y: number,
  maxDisparity: number,
  direction: -1 | 1,
  verticalOffset: number,
): Match | null {
  const w = source.width;
  const h = source.height;
  const sourceIndex = y * w + x;
  const costs = new Uint8Array(maxDisparity + 1);
  const ys = new Int16Array(maxDisparity + 1);
  costs.fill(255);
  let bestCost = 255;
  let bestD = 0;
  let bestY = 0;
  const expectedY = Math.round(y - verticalOffset);

  for (let d = 1; d <= maxDisparity; d++) {
    const tx = x + direction * d;
    if (tx < RADIUS || tx >= w - RADIUS) break;
    for (let ty = expectedY - 1; ty <= expectedY + 1; ty++) {
      if (ty < RADIUS || ty >= h - RADIUS) continue;
      const cost = distance(source, sourceIndex, target, ty * w + tx);
      if (cost < costs[d]) {
        costs[d] = cost;
        ys[d] = ty;
      }
    }
    if (costs[d] < bestCost) {
      bestCost = costs[d];
      bestD = d;
      bestY = ys[d];
    }
  }
  if (!bestD || bestCost > MAX_COST) return null;
  // Zero disparity is not a measurable depth, but it is still a competitor.
  // Ignoring it would invent positive distances for identical camera frames.
  let second = 255;
  for (let ty = expectedY - 1; ty <= expectedY + 1; ty++) {
    if (ty < RADIUS || ty >= h - RADIUS) continue;
    second = Math.min(second, distance(source, sourceIndex, target, ty * w + x));
  }
  for (let d = 1; d <= maxDisparity; d++) {
    if (Math.abs(d - bestD) <= 2) continue;
    if (costs[d] < second) second = costs[d];
  }
  if (second - bestCost < MIN_MARGIN) return null;
  return { x: x + direction * bestD, y: bestY, cost: bestCost };
}

const COLOR_STOPS = [
  [25, 67, 81],
  [29, 123, 137],
  [67, 186, 177],
  [157, 216, 164],
  [242, 222, 143],
];

/** Far is blue; near is pale yellow. */
export function depthColor(depth: number, maxDepth: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, 1 - (depth - 0.2) / Math.max(0.01, maxDepth - 0.2)));
  const scaled = t * (COLOR_STOPS.length - 1);
  const first = Math.min(COLOR_STOPS.length - 2, Math.floor(scaled));
  const f = scaled - first;
  return COLOR_STOPS[first].map((value, index) =>
    Math.round(value * (1 - f) + COLOR_STOPS[first + 1][index] * f),
  ) as [number, number, number];
}

/** Estimate a sparse, colourised 3D point cloud using positive disparity. */
export function reconstructStereo(leftFrame: Frame, rightFrame: Frame, settings: StereoSettings): PointCloudData {
  if (leftFrame.width !== rightFrame.width || leftFrame.height !== rightFrame.height) {
    throw new Error('Stereo frames must have the same size');
  }
  const { baselineCm, horizontalFov, maxDepthM } = settings;
  if (
    !Number.isFinite(baselineCm) || baselineCm <= 0 ||
    !Number.isFinite(horizontalFov) || horizontalFov <= 1 || horizontalFov >= 179 ||
    !Number.isFinite(maxDepthM) || maxDepthM <= 0.2 ||
    !Number.isFinite(settings.verticalOffset)
  ) {
    throw new Error('Invalid stereo settings');
  }
  const left = describe(leftFrame);
  const right = describe(rightFrame);
  const { width: w, height: h } = left;
  const focalPixels = w / (2 * Math.tan((horizontalFov * Math.PI) / 360));
  const baselineM = baselineCm / 100;
  const maxDisparity = Math.min(Math.floor(w * 0.32), Math.ceil((focalPixels * baselineM) / 0.2));
  const step = Math.max(2, Math.floor(settings.sampleStep ?? 3));
  const positions: number[] = [];
  const rgbColors: number[] = [];
  const depthColors: number[] = [];
  const depths: number[] = [];
  const pixelIndices: number[] = [];
  const map = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    map[i * 4] = 13;
    map[i * 4 + 1] = 31;
    map[i * 4 + 2] = 36;
    map[i * 4 + 3] = 255;
  }

  for (let y = RADIUS + 1; y < h - RADIUS - 1; y += step) {
    for (let x = RADIUS + 1; x < w - RADIUS - 1; x += step) {
      if (texture(left, x, y) < 24) continue;
      const forward = searchLine(left, right, x, y, maxDisparity, -1, settings.verticalOffset);
      if (!forward || texture(right, forward.x, forward.y) < 24) continue;
      const reverse = searchLine(right, left, forward.x, forward.y, maxDisparity, 1, -settings.verticalOffset);
      if (!reverse || Math.abs(reverse.x - x) > 1 || Math.abs(reverse.y - y) > 1) continue;
      const disparity = x - forward.x;
      const z = (focalPixels * baselineM) / disparity;
      if (z < 0.2 || z > maxDepthM) continue;
      const pointX = ((x - w / 2) * z) / focalPixels;
      const pointY = ((h / 2 - y) * z) / focalPixels;
      positions.push(pointX, pointY, -z);
      pixelIndices.push(y * w + x);
      const pixel = (y * w + x) * 4;
      rgbColors.push(leftFrame.data[pixel] / 255, leftFrame.data[pixel + 1] / 255, leftFrame.data[pixel + 2] / 255);
      const color = depthColor(z, maxDepthM);
      depthColors.push(color[0] / 255, color[1] / 255, color[2] / 255);
      depths.push(z);
      for (let yy = y; yy < Math.min(h, y + step); yy++) {
        for (let xx = x; xx < Math.min(w, x + step); xx++) {
          const p = (yy * w + xx) * 4;
          map[p] = color[0];
          map[p + 1] = color[1];
          map[p + 2] = color[2];
        }
      }
    }
  }

  depths.sort((a, b) => a - b);
  return {
    positions: new Float32Array(positions),
    rgbColors: new Float32Array(rgbColors),
    depthColors: new Float32Array(depthColors),
    depthMap: map,
    pixelIndices: new Uint32Array(pixelIndices),
    width: w,
    height: h,
    count: depths.length,
    near: depths.length ? depths[Math.floor((depths.length - 1) * 0.05)] : null,
    far: depths.length ? depths[Math.floor((depths.length - 1) * 0.95)] : null,
  };
}
