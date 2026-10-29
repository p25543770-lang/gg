/**
 * Corner detection (Shi-Tomasi "good features to track", the same family of
 * detectors OpenCV demos use to mark corners of calibration targets).
 *
 * For every pixel the local structure tensor of image gradients is built over a
 * 5×5 window; the smaller of its two eigenvalues measures "cornerness". Local
 * maxima above a quality threshold survive non-maximum suppression and greedy
 * spacing, and the UI draws them as small purple stroke marks — exactly on the
 * detected corners.
 */
import type { Frame } from './stereo';

export interface Corner {
  x: number;
  y: number;
  score: number;
}

/** Purple stroke marks for detected corners. */
export const CORNER_COLOR = '#a855f7';

export interface CornerOptions {
  maxCorners?: number;
  minDistance?: number;
  qualityLevel?: number;
}

const WINDOW_RADIUS = 2;

export function detectCorners(frame: Frame, options: CornerOptions = {}): Corner[] {
  const { width: w, height: h, data } = frame;
  const maxCorners = options.maxCorners ?? 60;
  const minDistance = options.minDistance ?? 7;
  const qualityLevel = options.qualityLevel ?? 0.04;
  if (w < 16 || h < 16 || data.length !== w * h * 4) return [];

  // Grayscale with the same luminance weights as the stereo matcher.
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = (77 * data[p] + 150 * data[p + 1] + 29 * data[p + 2]) / 256;
  }

  // Sobel gradients, normalised to intensity units.
  const ix = new Float32Array(w * h);
  const iy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        (gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1]) -
        (gray[i - w - 1] + 2 * gray[i - 1] + gray[i + w - 1]);
      const gy =
        (gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1]) -
        (gray[i - w - 1] + 2 * gray[i - w] + gray[i - w + 1]);
      ix[i] = gx / 4;
      iy[i] = gy / 4;
    }
  }

  // Structure tensor sums over the 5×5 window and the Shi-Tomasi score
  // (the smaller eigenvalue) at every candidate pixel.
  const scores = new Float32Array(w * h);
  let maxScore = 0;
  const margin = WINDOW_RADIUS + 1;
  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      let sxx = 0;
      let syy = 0;
      let sxy = 0;
      for (let dy = -WINDOW_RADIUS; dy <= WINDOW_RADIUS; dy++) {
        for (let dx = -WINDOW_RADIUS; dx <= WINDOW_RADIUS; dx++) {
          const j = (y + dy) * w + (x + dx);
          sxx += ix[j] * ix[j];
          syy += iy[j] * iy[j];
          sxy += ix[j] * iy[j];
        }
      }
      const window = (2 * WINDOW_RADIUS + 1) * (2 * WINDOW_RADIUS + 1);
      sxx /= window;
      syy /= window;
      sxy /= window;
      const trace = sxx + syy;
      const diff = sxx - syy;
      const discriminant = Math.sqrt(diff * diff + 4 * sxy * sxy);
      const minEigen = (trace - discriminant) / 2;
      const i = y * w + x;
      scores[i] = minEigen;
      if (minEigen > maxScore) maxScore = minEigen;
    }
  }
  if (maxScore <= 0) return [];

  const threshold = Math.max(220, qualityLevel * maxScore);
  const candidates: Corner[] = [];
  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      const score = scores[y * w + x];
      if (score < threshold) continue;
      // Local maximum in the 5×5 neighbourhood.
      let isPeak = true;
      for (let dy = -1; dy <= 1 && isPeak; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          if (scores[(y + dy) * w + (x + dx)] > score) {
            isPeak = false;
            break;
          }
        }
      }
      if (isPeak) candidates.push({ x, y, score });
    }
  }

  // Greedy selection by score with a minimum spacing so the marks stay readable.
  candidates.sort((a, b) => b.score - a.score);
  const picked: Corner[] = [];
  const minDistanceSq = minDistance * minDistance;
  for (const candidate of candidates) {
    let tooClose = false;
    for (const chosen of picked) {
      const dx = chosen.x - candidate.x;
      const dy = chosen.y - candidate.y;
      if (dx * dx + dy * dy < minDistanceSq) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) {
      picked.push(candidate);
      if (picked.length >= maxCorners) break;
    }
  }
  return picked;
}

export interface DrawStyle {
  /** Stroke length of each mark, in canvas pixels. */
  size?: number;
  color?: string;
  lineWidth?: number;
}

/**
 * Draw each corner as a small cross of two purple strokes ("полоски") — marks
 * sit exactly on the detected corners, no boxes or blobs.
 */
export function drawCornerMarks(
  context: CanvasRenderingContext2D,
  corners: Array<{ x: number; y: number }>,
  style: DrawStyle = {},
): void {
  const size = style.size ?? 4;
  const color = style.color ?? CORNER_COLOR;
  const lineWidth = style.lineWidth ?? 1.6;
  context.save();
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.lineCap = 'round';
  context.shadowColor = 'rgba(20, 8, 40, 0.55)';
  context.shadowBlur = 2;
  for (const corner of corners) {
    context.beginPath();
    context.moveTo(corner.x - size, corner.y);
    context.lineTo(corner.x + size, corner.y);
    context.moveTo(corner.x, corner.y - size);
    context.lineTo(corner.x, corner.y + size);
    context.stroke();
  }
  context.restore();
}
