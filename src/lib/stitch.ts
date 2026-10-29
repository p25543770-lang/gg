/**
 * Geometric merging of two side-by-side cameras into one combined live frame.
 *
 * The two views are reprojected into a virtual camera placed halfway between
 * the physical ones ("mid view"). For a rectified-ish pair a 3D point with
 * disparity d = x_left - x_right projects to x_mid = x_left - d/2 = x_right +
 * d/2, so each pixel is shifted by half its OWN disparity. This compensates
 * parallax at the depth of that pixel — near and far objects land where they
 * belong instead of doubling the way a single global homography would.
 *
 * Disparity comes from mutual census block matching (the same conservative
 * matcher the rest of the app uses). Where disparity is NOT reliable the code
 * deliberately degrades to a small global shift instead of trusting bad depth,
 * and the overlap is taken from one camera rather than blended into a double
 * silhouette. Regions no camera saw stay empty and are flagged — they are
 * never filled with invented content.
 */
import {
  describe,
  median,
  searchLine,
  texture,
  type Frame,
} from './stereo';

export interface DenseDisparity {
  width: number;
  height: number;
  /** x_left - x_right per pixel of the left frame. 0 where unknown. */
  disparity: Float32Array;
  /** 0..255 per pixel; derived from real match support. */
  confidence: Uint8Array;
  /** The same field expressed in right-frame coordinates. */
  disparityRight: Float32Array;
  confidenceRight: Uint8Array;
  /** Grid matches consistent with their local disparity (spatially coherent). */
  inlierMatches: Array<{ x: number; y: number; xr: number; yr: number; d: number }>;
  /** Number of spatially coherent matches — random noise matches are dropped. */
  matchCount: number;
  /** Median disparity of confident matches. */
  median: number;
  /** Interquartile spread of confident matches. */
  spread: number;
  /** Fraction of pixels with confident disparity, 0..1. */
  confidentRatio: number;
}

export type CombineMode = 'depth' | 'planar';

export interface ViewMappingDepth {
  kind: 'depth';
  width: number;
  height: number;
  /** Output x = source x - d/2 + shift (left frame), + d/2 + shift (right). */
  shift: number;
  disparity: Float32Array;
  confidence: Uint8Array;
}

export interface ViewMappingPlanar {
  kind: 'planar';
  width: number;
  height: number;
  /** Right frame is warped into the left frame by this homography. */
  homography: number[];
}

export type ViewMapping = ViewMappingDepth | ViewMappingPlanar;

export interface CombinedFrame {
  width: number;
  height: number;
  data: Uint8ClampedArray;
  /** 0 = no camera saw this pixel; 255 = full photographic data. */
  coverage: Uint8ClampedArray;
  mode: CombineMode;
  mapping: ViewMapping;
  /** Fraction of the output area covered by at least one camera. */
  fillRatio: number;
  /** Fraction of the overlap that used parallax-correct reprojection. */
  parallaxRatio: number;
  photometric: { gain: [number, number, number]; bias: [number, number, number] };
}

export interface MappedQuad {
  /** Eight outline points (corners + edge midpoints) in combined pixels. */
  quad: Array<{ x: number; y: number }>;
  bbox: { left: number; top: number; right: number; bottom: number };
  /** True when the warp at the box is backed by measured disparity. */
  confident: boolean;
}

const RADIUS = 3;
const TILE = 16;
const MIN_MATCH_TEXTURE = 24;
const MAX_MATCH_COST = 10;

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value));
}

export function combineShift(width: number, maxDisparity: number): number {
  return Math.ceil(Math.min(Math.max(maxDisparity, 4), Math.floor(width * 0.3)) / 2) + 2;
}

/** Grid-based census matching with mutual consistency checks. */
export function matchGrid(
  left: Frame,
  right: Frame,
  verticalOffset: number,
  maxDisparity: number,
  step = 2,
): Array<{ x: number; y: number; xr: number; yr: number; d: number }> {
  const a = describe(left);
  const b = describe(right);
  const matches: Array<{ x: number; y: number; xr: number; yr: number; d: number }> = [];
  const maxD = Math.min(Math.max(maxDisparity, 4), Math.floor(a.width * 0.3));
  for (let y = RADIUS + 1; y < a.height - RADIUS - 1; y += step) {
    for (let x = RADIUS + 1; x < a.width - RADIUS - 1; x += step) {
      if (texture(a, x, y) < MIN_MATCH_TEXTURE) continue;
      const forward = searchLine(a, b, x, y, maxD, -1, verticalOffset);
      // A stricter cost than the sparse stereo matcher: an accidental census
      // coincidence of unrelated pixels must not become "measured geometry".
      if (!forward || forward.cost > MAX_MATCH_COST || texture(b, forward.x, forward.y) < MIN_MATCH_TEXTURE) continue;
      const reverse = searchLine(b, a, forward.x, forward.y, maxD, 1, -verticalOffset);
      if (!reverse || reverse.cost > MAX_MATCH_COST || Math.abs(reverse.x - x) > 1 || Math.abs(reverse.y - y) > 1) continue;
      const d = x - forward.x;
      if (d < 1) continue;
      matches.push({ x, y, xr: forward.x, yr: forward.y, d });
    }
  }
  return matches;
}

interface TileField {
  d: Float32Array;
  count: Uint16Array;
  spread: Float32Array;
  tilesX: number;
  tilesY: number;
}

function tileField(
  points: Array<{ x: number; y: number; d: number }>,
  width: number,
  height: number,
  globalMedian: number,
): TileField {
  const tilesX = Math.ceil(width / TILE);
  const tilesY = Math.ceil(height / TILE);
  const d = new Float32Array(tilesX * tilesY).fill(globalMedian);
  const count = new Uint16Array(tilesX * tilesY);
  const spread = new Float32Array(tilesX * tilesY).fill(Infinity);
  const buckets: number[][] = Array.from({ length: tilesX * tilesY }, () => []);
  for (const point of points) {
    const tx = clamp(Math.floor(point.x / TILE), 0, tilesX - 1);
    const ty = clamp(Math.floor(point.y / TILE), 0, tilesY - 1);
    buckets[ty * tilesX + tx].push(point.d);
  }
  for (let i = 0; i < buckets.length; i++) {
    const values = buckets[i];
    if (!values.length) continue;
    d[i] = median(values);
    count[i] = values.length;
    if (values.length >= 4) {
      const sorted = [...values].sort((x, y) => x - y);
      spread[i] = sorted[Math.floor((sorted.length - 1) * 0.75)] - sorted[Math.floor((sorted.length - 1) * 0.25)];
    } else {
      spread[i] = 0;
    }
  }
  // Propagate into empty tiles from the nearest filled tile (limited radius).
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const index = ty * tilesX + tx;
      if (count[index]) continue;
      let best = -1;
      let bestDist = Infinity;
      for (let radius = 1; radius <= 3; radius++) {
        for (let ny = Math.max(0, ty - radius); ny <= Math.min(tilesY - 1, ty + radius); ny++) {
          for (let nx = Math.max(0, tx - radius); nx <= Math.min(tilesX - 1, tx + radius); nx++) {
            const other = ny * tilesX + nx;
            if (!count[other]) continue;
            const dist = Math.abs(nx - tx) + Math.abs(ny - ty);
            if (dist < bestDist) {
              bestDist = dist;
              best = other;
            }
          }
        }
        if (best >= 0) break;
      }
      if (best >= 0) {
        d[index] = d[best];
        count[index] = 0;
        spread[index] = Infinity;
      }
    }
  }
  return { d, count, spread, tilesX, tilesY };
}

function sampleTileField(field: TileField, x: number, y: number): { d: number; confidence: number } {
  const { tilesX, tilesY, d, count, spread } = field;
  const fx = clamp(x / TILE - 0.5, 0, tilesX - 1);
  const fy = clamp(y / TILE - 0.5, 0, tilesY - 1);
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(tilesX - 1, x0 + 1);
  const y1 = Math.min(tilesY - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  let total = 0;
  let weighted = 0;
  let confidence = 0;
  const corners: Array<[number, number, number]> = [
    [x0, y0, (1 - tx) * (1 - ty)],
    [x1, y0, tx * (1 - ty)],
    [x0, y1, (1 - tx) * ty],
    [x1, y1, tx * ty],
  ];
  for (const [cx, cy, weight] of corners) {
    const index = cy * tilesX + cx;
    weighted += d[index] * weight;
    total += weight;
    const support = clamp(count[index] / 4, 0, 1);
    const coherence = spread[index] === Infinity ? 0 : clamp(1 - spread[index] / 8, 0, 1);
    confidence = Math.max(confidence, weight * (0.35 * support + 0.65 * coherence) * (count[index] ? 1 : 0.4));
  }
  return { d: total > 0 ? weighted / total : 0, confidence: clamp(confidence, 0, 1) };
}

/** Dense per-pixel disparity with explicit confidence; unknown stays at 0. */
export function buildDisparityField(
  left: Frame,
  right: Frame,
  verticalOffset: number,
  maxDisparity: number,
): DenseDisparity {
  const matches = matchGrid(left, right, verticalOffset, maxDisparity);
  const positive = matches.filter((m) => m.d >= 1).map((m) => m.d);
  const globalMedian = positive.length ? median(positive) : 0;

  // Keep only matches coherent with their local tile median: scattered random
  // matches of unrelated scenes must not count as measured geometry.
  const tilesX = Math.ceil(left.width / TILE);
  const tilesY = Math.ceil(left.height / TILE);
  const buckets: number[][] = Array.from({ length: tilesX * tilesY }, () => []);
  for (const m of matches) {
    const tx = clamp(Math.floor(m.x / TILE), 0, tilesX - 1);
    const ty = clamp(Math.floor(m.y / TILE), 0, tilesY - 1);
    buckets[ty * tilesX + tx].push(m.d);
  }
  const tileMedians = buckets.map((values) => (values.length ? median(values) : globalMedian));
  const inlierMatches = matches.filter((m) => {
    const tx = clamp(Math.floor(m.x / TILE), 0, tilesX - 1);
    const ty = clamp(Math.floor(m.y / TILE), 0, tilesY - 1);
    return Math.abs(m.d - tileMedians[ty * tilesX + tx]) <= 3;
  });

  const sorted = inlierMatches.map((m) => m.d).sort((a, b) => a - b);
  const spread = sorted.length >= 4
    ? sorted[Math.floor((sorted.length - 1) * 0.75)] - sorted[Math.floor((sorted.length - 1) * 0.25)]
    : 0;

  const leftField = tileField(inlierMatches, left.width, left.height, globalMedian);
  const rightField = tileField(
    inlierMatches.map((m) => ({ x: m.xr, y: m.yr, d: m.d })),
    right.width,
    right.height,
    globalMedian,
  );

  const disparity = new Float32Array(left.width * left.height);
  const confidence = new Uint8Array(left.width * left.height);
  const disparityRight = new Float32Array(right.width * right.height);
  const confidenceRight = new Uint8Array(right.width * right.height);
  let confident = 0;
  for (let y = 0; y < left.height; y++) {
    for (let x = 0; x < left.width; x++) {
      const sample = sampleTileField(leftField, x, y);
      const index = y * left.width + x;
      disparity[index] = sample.d;
      confidence[index] = Math.round(sample.confidence * 255);
      if (sample.confidence >= 0.5) confident++;
    }
  }
  for (let y = 0; y < right.height; y++) {
    for (let x = 0; x < right.width; x++) {
      const sample = sampleTileField(rightField, x, y);
      const index = y * right.width + x;
      disparityRight[index] = sample.d;
      confidenceRight[index] = Math.round(sample.confidence * 255);
    }
  }

  return {
    width: left.width,
    height: left.height,
    disparity,
    confidence,
    disparityRight,
    confidenceRight,
    inlierMatches,
    matchCount: inlierMatches.length,
    median: sorted.length ? sorted[Math.floor((sorted.length - 1) * 0.5)] : 0,
    spread,
    confidentRatio: left.width * left.height ? confident / (left.width * left.height) : 0,
  };
}

/**
 * Per-channel gain and bias that bring the right frame onto the left frame's
 * exposure, fitted on pixels the matcher already agreed about.
 */
export function photometricCompensation(
  left: Frame,
  right: Frame,
  matches: Array<{ x: number; y: number; xr: number; yr: number }>,
): { gain: [number, number, number]; bias: [number, number, number] } {
  const gain: [number, number, number] = [1, 1, 1];
  const bias: [number, number, number] = [0, 0, 0];
  const usable = matches.slice(0, 400);
  if (usable.length < 12) return { gain, bias };
  for (let c = 0; c < 3; c++) {
    let sumL = 0;
    let sumR = 0;
    for (const m of usable) {
      sumL += left.data[(m.y * left.width + m.x) * 4 + c];
      sumR += right.data[(m.yr * right.width + m.xr) * 4 + c];
    }
    const meanL = sumL / usable.length;
    const meanR = sumR / usable.length;
    if (meanR < 8) continue;
    let varR = 0;
    let cov = 0;
    for (const m of usable) {
      const l = left.data[(m.y * left.width + m.x) * 4 + c];
      const r = right.data[(m.yr * right.width + m.xr) * 4 + c];
      varR += (r - meanR) * (r - meanR);
      cov += (r - meanR) * (l - meanL);
    }
    const slope = varR > 1 ? clamp(cov / varR, 0.6, 1.7) : clamp(meanL / meanR, 0.6, 1.7);
    gain[c] = slope;
    bias[c] = clamp(meanL - slope * meanR, -64, 64);
  }
  return { gain, bias };
}

function splat(
  target: Float32Array,
  weights: Float32Array,
  outWidth: number,
  outHeight: number,
  x: number,
  y: number,
  r: number,
  g: number,
  b: number,
) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  for (let dy = 0; dy <= 1; dy++) {
    const py = iy + dy;
    if (py < 0 || py >= outHeight) continue;
    const wy = dy ? fy : 1 - fy;
    if (wy <= 0) continue;
    for (let dx = 0; dx <= 1; dx++) {
      const px = ix + dx;
      if (px < 0 || px >= outWidth) continue;
      const w = wy * (dx ? fx : 1 - fx);
      if (w <= 0) continue;
      const index = (py * outWidth + px) * 3;
      target[index] += r * w;
      target[index + 1] += g * w;
      target[index + 2] += b * w;
      weights[py * outWidth + px] += w;
    }
  }
}

const smoothstep = (t: number) => {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
};

/**
 * Reproject both frames into the mid viewpoint and blend the overlap.
 * Confident disparity → parallax-correct shift per pixel. Doubtful disparity →
 * a small global shift only (never a big warp from bad depth) and a single
 * source in the overlap (never a double silhouette).
 */
export function combineFrames(
  left: Frame,
  right: Frame,
  field: DenseDisparity,
  options: { maxDisparity: number; verticalOffset: number },
): CombinedFrame {
  const { width: w, height: h } = left;
  const shift = combineShift(w, options.maxDisparity);
  const outWidth = w + 2 * shift;
  const outHeight = h;
  const matches = matchGrid(left, right, options.verticalOffset, options.maxDisparity, 4);
  const photometric = photometricCompensation(left, right, matches);
  const fallbackD = field.median;

  const sumL = new Float32Array(outWidth * outHeight * 3);
  const wL = new Float32Array(outWidth * outHeight);
  const sumR = new Float32Array(outWidth * outHeight * 3);
  const wR = new Float32Array(outWidth * outHeight);
  const confL = new Float32Array(outWidth * outHeight);
  const confR = new Float32Array(outWidth * outHeight);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const index = y * w + x;
      const p = index * 4;
      const conf = field.confidence[index] / 255;
      const d = conf >= 0.5 ? field.disparity[index] : fallbackD;
      const outX = x - d / 2 + shift;
      splat(sumL, wL, outWidth, outHeight, outX, y, left.data[p], left.data[p + 1], left.data[p + 2]);
      const nearestX = clamp(Math.round(outX), 0, outWidth - 1);
      confL[y * outWidth + nearestX] = Math.max(confL[y * outWidth + nearestX], conf);

      const rIndex = y * w + x;
      const rp = rIndex * 4;
      const confRPixel = field.confidenceRight[rIndex] / 255;
      const dR = confRPixel >= 0.5 ? field.disparityRight[rIndex] : fallbackD;
      const outXR = x + dR / 2 + shift;
      const r = clamp(right.data[rp] * photometric.gain[0] + photometric.bias[0], 0, 255);
      const g = clamp(right.data[rp + 1] * photometric.gain[1] + photometric.bias[1], 0, 255);
      const b = clamp(right.data[rp + 2] * photometric.gain[2] + photometric.bias[2], 0, 255);
      splat(sumR, wR, outWidth, outHeight, outXR, y, r, g, b);
      const nearestXR = clamp(Math.round(outXR), 0, outWidth - 1);
      confR[y * outWidth + nearestXR] = Math.max(confR[y * outWidth + nearestXR], confRPixel);
    }
  }

  const data = new Uint8ClampedArray(outWidth * outHeight * 4);
  const coverage = new Uint8ClampedArray(outWidth * outHeight);
  let filled = 0;
  let parallaxPixels = 0;
  let overlapPixels = 0;

  for (let y = 0; y < outHeight; y++) {
    // Overlap band of this row, used for the smooth transition.
    let o0 = -1;
    let o1 = -1;
    for (let x = 0; x < outWidth; x++) {
      const i = y * outWidth + x;
      if (wL[i] > 0.02 && wR[i] > 0.02) {
        if (o0 < 0) o0 = x;
        o1 = x;
      }
    }
    for (let x = 0; x < outWidth; x++) {
      const i = y * outWidth + x;
      const hasL = wL[i] > 0.02;
      const hasR = wR[i] > 0.02;
      const out = i * 4;
      if (!hasL && !hasR) {
        data[out + 3] = 255;
        continue;
      }
      let r: number;
      let g: number;
      let b: number;
      if (hasL && hasR) {
        overlapPixels++;
        const lr = sumL[i * 3] / wL[i];
        const lg = sumL[i * 3 + 1] / wL[i];
        const lb = sumL[i * 3 + 2] / wL[i];
        const rr = sumR[i * 3] / wR[i];
        const rg = sumR[i * 3 + 1] / wR[i];
        const rb = sumR[i * 3 + 2] / wR[i];
        const conf = Math.max(confL[i], confR[i]);
        if (conf < 0.35) {
          // Doubtful geometry: keep one real image instead of blending ghosts.
          r = lr;
          g = lg;
          b = lb;
        } else {
          parallaxPixels++;
          const t = o1 > o0 ? smoothstep((x - o0) / (o1 - o0)) : 0.5;
          r = lr * (1 - t) + rr * t;
          g = lg * (1 - t) + rg * t;
          b = lb * (1 - t) + rb * t;
        }
        coverage[i] = 255;
      } else if (hasL) {
        r = sumL[i * 3] / wL[i];
        g = sumL[i * 3 + 1] / wL[i];
        b = sumL[i * 3 + 2] / wL[i];
        coverage[i] = 255;
      } else {
        r = sumR[i * 3] / wR[i];
        g = sumR[i * 3 + 1] / wR[i];
        b = sumR[i * 3 + 2] / wR[i];
        coverage[i] = 255;
      }
      data[out] = r;
      data[out + 1] = g;
      data[out + 2] = b;
      data[out + 3] = 255;
      filled++;
    }
  }

  return {
    width: outWidth,
    height: outHeight,
    data,
    coverage,
    mode: 'depth',
    mapping: {
      kind: 'depth',
      width: w,
      height: h,
      shift,
      disparity: field.disparity,
      confidence: field.confidence,
    },
    fillRatio: filled / (outWidth * outHeight),
    parallaxRatio: overlapPixels ? parallaxPixels / overlapPixels : 0,
    photometric,
  };
}

/** Planar fallback: fit a real homography to real correspondences (RANSAC). */
export function estimateHomography(
  points: Array<{ from: { x: number; y: number }; to: { x: number; y: number } }>,
  iterations = 120,
  threshold = 3,
): { h: number[]; inliers: number } | null {
  if (points.length < 8) return null;
  const solve4 = (sample: typeof points): number[] | null => {
    const rows: number[][] = [];
    for (const { from, to } of sample) {
      rows.push([from.x, from.y, 1, 0, 0, 0, -to.x * from.x, -to.x * from.y, to.x]);
      rows.push([0, 0, 0, from.x, from.y, 1, -to.y * from.x, -to.y * from.y, to.y]);
    }
    // Solve the 8x8 system (h22 normalised to 1) by Gaussian elimination.
    const a = rows.map((row) => row.slice(0, 8));
    const b = rows.map((row) => row[8]);
    const n = 8;
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let row = col + 1; row < n; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
      if (Math.abs(a[pivot][col]) < 1e-9) return null;
      [a[col], a[pivot]] = [a[pivot], a[col]];
      [b[col], b[pivot]] = [b[pivot], b[col]];
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = a[row][col] / a[col][col];
        for (let k = col; k < n; k++) a[row][k] -= factor * a[col][k];
        b[row] -= factor * b[col];
      }
    }
    const h = b.map((value, i) => value / a[i][i]);
    return [...h, 1];
  };
  const apply = (h: number[], x: number, y: number) => {
    const w = h[6] * x + h[7] * y + h[8];
    return Math.abs(w) < 1e-9 ? null : { x: (h[0] * x + h[1] * y + h[2]) / w, y: (h[3] * x + h[4] * y + h[5]) / w };
  };
  const error = (h: number[], p: (typeof points)[number]) => {
    const mapped = apply(h, p.from.x, p.from.y);
    return mapped ? Math.hypot(mapped.x - p.to.x, mapped.y - p.to.y) : Infinity;
  };

  let best: number[] | null = null;
  let bestInliers = 0;
  let seed = 20261029;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let iteration = 0; iteration < iterations; iteration++) {
    const sample = [];
    const used = new Set<number>();
    while (sample.length < 4) {
      const pick = Math.floor(random() * points.length);
      if (used.has(pick)) continue;
      used.add(pick);
      sample.push(points[pick]);
    }
    const h = solve4(sample);
    if (!h) continue;
    const inliers = points.filter((p) => error(h, p) < threshold);
    if (inliers.length > bestInliers) {
      bestInliers = inliers.length;
      best = h;
    }
  }
  if (!best || bestInliers < 12 || bestInliers / points.length < 0.35) return null;
  // Refine on inliers via normal equations of the (over-determined) system.
  const inlierPoints = points.filter((p) => error(best!, p) < threshold).slice(0, 240);
  const ata = Array.from({ length: 8 }, () => new Float64Array(8));
  const atb = new Float64Array(8);
  for (const { from, to } of inlierPoints) {
    const row = [from.x, from.y, 1, 0, 0, 0, -to.x * from.x, -to.x * from.y];
    const row2 = [0, 0, 0, from.x, from.y, 1, -to.y * from.x, -to.y * from.y];
    const targets = [to.x, to.y];
    [row, row2].forEach((r, idx) => {
      for (let i = 0; i < 8; i++) {
        atb[i] += r[i] * targets[idx];
        for (let j = 0; j < 8; j++) ata[i][j] += r[i] * r[j];
      }
    });
  }
  for (let col = 0; col < 8; col++) {
    let pivot = col;
    for (let row = col + 1; row < 8; row++) if (Math.abs(ata[row][col]) > Math.abs(ata[pivot][col])) pivot = row;
    if (Math.abs(ata[pivot][col]) < 1e-9) return { h: best, inliers: bestInliers };
    const tmpRow = ata[col];
    ata[col] = ata[pivot];
    ata[pivot] = tmpRow;
    const tmpB = atb[col];
    atb[col] = atb[pivot];
    atb[pivot] = tmpB;
    for (let row = 0; row < 8; row++) {
      if (row === col) continue;
      const factor = ata[row][col] / ata[col][col];
      for (let k = col; k < 8; k++) ata[row][k] -= factor * ata[col][k];
      atb[row] -= factor * atb[col];
    }
  }
  const refined = [...atb.map((value, i) => value / ata[i][i]), 1];
  return { h: refined, inliers: bestInliers };
}

/**
 * Planar/panoramic mode: the right frame is warped into the LEFT frame by the
 * fitted homography and feathered over the overlap. Only valid for mostly
 * planar scenes — call sites must label it and its limits.
 */
export function combinePlanar(
  left: Frame,
  right: Frame,
  homography: number[],
  photometric: { gain: [number, number, number]; bias: [number, number, number] },
): CombinedFrame {
  const { width: w, height: h } = left;
  const sumR = new Float32Array(w * h * 3);
  const wR = new Float32Array(w * h);
  const apply = (x: number, y: number) => {
    const denom = homography[6] * x + homography[7] * y + homography[8];
    if (Math.abs(denom) < 1e-9) return null;
    return {
      x: (homography[0] * x + homography[1] * y + homography[2]) / denom,
      y: (homography[3] * x + homography[4] * y + homography[5]) / denom,
    };
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const mapped = apply(x, y);
      if (!mapped) continue;
      const sx = mapped.x;
      const sy = mapped.y;
      if (sx < -1 || sy < -1 || sx >= w || sy >= h) continue;
      const ix = clamp(Math.floor(sx), 0, w - 1);
      const iy = clamp(Math.floor(sy), 0, h - 1);
      const p = (iy * w + ix) * 4;
      const r = clamp(right.data[p] * photometric.gain[0] + photometric.bias[0], 0, 255);
      const g = clamp(right.data[p + 1] * photometric.gain[1] + photometric.bias[1], 0, 255);
      const b = clamp(right.data[p + 2] * photometric.gain[2] + photometric.bias[2], 0, 255);
      splat(sumR, wR, w, h, sx, sy, r, g, b);
    }
  }
  const data = new Uint8ClampedArray(w * h * 4);
  const coverage = new Uint8ClampedArray(w * h);
  let filled = 0;
  let overlap = 0;
  for (let y = 0; y < h; y++) {
    let o0 = -1;
    let o1 = -1;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (wR[i] > 0.02) {
        if (o0 < 0) o0 = x;
        o1 = x;
      }
    }
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const out = i * 4;
      const hasR = wR[i] > 0.02;
      const p = i * 4;
      const lr = left.data[p];
      const lg = left.data[p + 1];
      const lb = left.data[p + 2];
      if (!hasR) {
        data[out] = lr;
        data[out + 1] = lg;
        data[out + 2] = lb;
        data[out + 3] = 255;
        coverage[i] = 255;
        filled++;
        continue;
      }
      overlap++;
      const rr = sumR[i * 3] / wR[i];
      const rg = sumR[i * 3 + 1] / wR[i];
      const rb = sumR[i * 3 + 2] / wR[i];
      // Feather the transition inside the overlap band only.
      const t = o1 > o0 ? smoothstep((x - o0) / (o1 - o0)) : 0.5;
      data[out] = lr * (1 - t) + rr * t;
      data[out + 1] = lg * (1 - t) + rg * t;
      data[out + 2] = lb * (1 - t) + rb * t;
      data[out + 3] = 255;
      coverage[i] = 255;
      filled++;
    }
  }
  return {
    width: w,
    height: h,
    data,
    coverage,
    mode: 'planar',
    mapping: { kind: 'planar', width: w, height: h, homography },
    fillRatio: filled / (w * h),
    parallaxRatio: 0,
    photometric,
  };
}

/**
 * Transfer a single point from the left frame through the SAME warp the image
 * went through (half its measured disparity towards the mid viewpoint).
 */
export function transferPointToCombined(point: { x: number; y: number }, mapping: ViewMapping): { x: number; y: number } {
  if (mapping.kind === 'planar') return { ...point };
  const w = mapping.width;
  const h = mapping.height;
  const x = clamp(Math.round(point.x), 0, w - 1);
  const y = clamp(Math.round(point.y), 0, h - 1);
  const index = y * w + x;
  const d = mapping.confidence[index] / 255 >= 0.5 ? mapping.disparity[index] : 0;
  return { x: point.x - d / 2 + mapping.shift, y: point.y };
}

/**
 * Transfer a detection box from the left frame through the SAME warp the
 * image went through. Outline points are mapped individually so the quad
 * follows depth changes instead of pretending the box is a rigid rectangle.
 */
export function transferBoxToCombined(box: { left: number; top: number; right: number; bottom: number }, mapping: ViewMapping): MappedQuad {
  const w = mapping.width;
  const h = mapping.height;
  const corners = [
    { x: box.left * w, y: box.top * h },
    { x: box.right * w, y: box.top * h },
    { x: box.right * w, y: box.bottom * h },
    { x: box.left * w, y: box.bottom * h },
    { x: (box.left + box.right) / 2 * w, y: box.top * h },
    { x: box.right * w, y: (box.top + box.bottom) / 2 * h },
    { x: (box.left + box.right) / 2 * w, y: box.bottom * h },
    { x: box.left * w, y: (box.top + box.bottom) / 2 * h },
  ];
  const quad = corners.map((point) => transferPointToCombined(point, mapping));
  const xs = quad.map((p) => p.x);
  const ys = quad.map((p) => p.y);
  let conf = 0;
  if (mapping.kind === 'depth') {
    for (const point of corners) {
      const x = clamp(Math.round(point.x), 0, w - 1);
      const y = clamp(Math.round(point.y), 0, h - 1);
      conf += mapping.confidence[y * w + x] / 255;
    }
    conf /= corners.length;
  } else {
    conf = 1;
  }
  return {
    quad,
    bbox: { left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys) },
    confident: conf >= 0.5,
  };
}

export interface SetupAssessment {
  ready: boolean;
  reason?: 'few-matches' | 'low-overlap' | 'vertical-misalignment' | 'no-texture' | 'inconsistent' | 'disconnected';
  warnings: string[];
}

/** Decide whether this pair can produce a trustworthy combined view. */
export function evaluateSetup(
  result: {
    matches: number;
    agreement: number;
    verticalOffset: number;
    overlapRatio: number;
    sharpnessA: number;
    sharpnessB: number;
    brightnessRatio: number;
    order: string | null;
  },
  frameHeight: number,
): SetupAssessment {
  const warnings: string[] = [];
  if (!result.order) {
    if (result.matches < 8) return { ready: false, reason: 'few-matches', warnings };
    return { ready: false, reason: 'inconsistent', warnings };
  }
  if (result.matches < 12) return { ready: false, reason: 'few-matches', warnings };
  if (result.overlapRatio < 0.35) return { ready: false, reason: 'low-overlap', warnings };
  if (Math.abs(result.verticalOffset) > frameHeight * 0.08) return { ready: false, reason: 'vertical-misalignment', warnings };
  if (result.sharpnessA < 10 || result.sharpnessB < 10) return { ready: false, reason: 'no-texture', warnings };
  if (result.brightnessRatio > 2.2 || result.brightnessRatio < 1 / 2.2) warnings.push('exposure');
  if (result.overlapRatio < 0.55) warnings.push('small-overlap');
  return { ready: true, warnings };
}

export function describeSetupReason(reason: SetupAssessment['reason']): { title: string; advice: string } {
  switch (reason) {
    case 'few-matches':
      return { title: 'Недостаточно общих деталей', advice: 'Добавьте в общую область текстурированные объекты и улучшите освещение.' };
    case 'low-overlap':
      return { title: 'Недостаточно перекрытия', advice: 'Сведите камеры ближе к общей сцене или разверните их друг к другу.' };
    case 'vertical-misalignment':
      return { title: 'Камеры смещены по высоте', advice: 'Выровняйте камеры по высоте и закрепите их — затем повторите настройку.' };
    case 'no-texture':
      return { title: 'Слишком мало деталей в кадре', advice: 'Направьте камеры на сцену с заметными деталями, а не на однотонную поверхность.' };
    case 'inconsistent':
      return { title: 'Кадры камер не согласуются', advice: 'Проверьте, что обе камеры смотрят на одну сцену, и выберите порядок камер вручную.' };
    case 'disconnected':
      return { title: 'Камера отключена', advice: 'Подключите обе камеры заново и повторите настройку.' };
    default:
      return { title: 'Настройка не завершена', advice: 'Нажмите «Автонастройка» ещё раз после проверки камер.' };
  }
}
