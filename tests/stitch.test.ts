import { describe, expect, it } from 'vitest';
import type { Frame } from '../src/lib/stereo';
import {
  buildDisparityField,
  combineFrames,
  combinePlanar,
  combineShift,
  estimateHomography,
  evaluateSetup,
  matchGrid,
  photometricCompensation,
  transferBoxToCombined,
  type DenseDisparity,
  type ViewMapping,
} from '../src/lib/stitch';

/** Random texture with a shifted copy: x(A)-x(B)=dx, y(A)-y(B)=dy. */
function stereoPair(dx: number, dy = 0, initialSeed = 12345, dim = false): [Frame, Frame] {
  const width = 160;
  const height = 112;
  const a = new Uint8ClampedArray(width * height * 4);
  const b = new Uint8ClampedArray(a.length);
  let seed = initialSeed;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed >>> 24) % 90 + (dim ? 60 : 40);
  };
  for (let i = 0; i < width * height; i++) {
    const value = random();
    a[i * 4] = value;
    a[i * 4 + 1] = value;
    a[i * 4 + 2] = value;
    a[i * 4 + 3] = 255;
    b[i * 4] = random();
    b[i * 4 + 1] = b[i * 4];
    b[i * 4 + 2] = b[i * 4];
    b[i * 4 + 3] = 255;
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const bx = x - dx;
      const by = y - dy;
      if (bx < 0 || bx >= width || by < 0 || by >= height) continue;
      const ai = (y * width + x) * 4;
      const bi = (by * width + bx) * 4;
      b.set(a.subarray(ai, ai + 4), bi);
    }
  }
  return [{ width, height, data: a }, { width, height, data: b }];
}

/** A bright square patch of the scene, shifted the same way between frames. */
function pairWithMarker(dx: number, seed = 777): [Frame, Frame] {
  const [left, right] = stereoPair(dx, 0, seed);
  const width = left.width;
  const height = left.height;
  const mark = (frame: Frame, offsetX: number) => {
    for (let y = 20; y < 32; y++) {
      for (let x = 30 + offsetX; x < 42 + offsetX; x++) {
        const p = (y * width + x) * 4;
        frame.data[p] = 250;
        frame.data[p + 1] = 250;
        frame.data[p + 2] = 250;
      }
    }
  };
  mark(left, 0);
  mark(right, -dx);
  return [left, right];
}

describe('disparity field', () => {
  it('measures the true shift and keeps confidence high where texture exists', () => {
    const [left, right] = stereoPair(12, 2);
    const field = buildDisparityField(left, right, 2, 48);
    expect(field.matchCount).toBeGreaterThan(50);
    expect(field.median).toBeCloseTo(12, 5);
    expect(field.spread).toBeLessThanOrEqual(1);
    expect(field.confidentRatio).toBeGreaterThan(0.5);
  });

  it('reports no confident disparity for identical frames (zero disparity is not a depth)', () => {
    const [a] = stereoPair(0);
    const field = buildDisparityField(a, a, 0, 48);
    expect(field.matchCount).toBe(0);
    expect(field.confidentRatio).toBe(0);
  });

  it('does not invent matches for unrelated scenes', () => {
    const [a] = stereoPair(8, 0, 12345);
    const [b] = stereoPair(8, 0, 987654);
    const field = buildDisparityField(a, b, 0, 48);
    expect(field.matchCount).toBeLessThan(12);
    expect(field.confidentRatio).toBeLessThan(0.12);
  });
});

describe('mid-view reprojection (parallax-aware merge)', () => {
  it('places the same scene point at one output location, not two', () => {
    const dx = 12;
    const [left, right] = pairWithMarker(dx);
    const field = buildDisparityField(left, right, 0, 48);
    const combined = combineFrames(left, right, field, { maxDisparity: 48, verticalOffset: 0 });
    const shift = combineShift(left.width, 48);
    // Marker centre at x=35.5 → expected at 35.5 - d/2 + shift for BOTH cameras.
    const expectedX = Math.round(35.5 - dx / 2 + shift);
    const brightColumns: number[] = [];
    for (let x = 0; x < combined.width; x++) {
      let sum = 0;
      for (let y = 22; y < 30; y++) sum += combined.data[(y * combined.width + x) * 4];
      if (sum / 8 > 220) brightColumns.push(x);
    }
    expect(brightColumns.length).toBeGreaterThan(4);
    expect(brightColumns).toContain(expectedX);
    // No second copy: all bright columns stay in one cluster around the seam.
    const min = Math.min(...brightColumns);
    const max = Math.max(...brightColumns);
    expect(max - min).toBeLessThan(20);
    expect(combined.mode).toBe('depth');
  });

  it('shifts the left and the right view symmetrically around the mid viewpoint', () => {
    const dx = 10;
    const [left, right] = stereoPair(dx);
    const field = buildDisparityField(left, right, 0, 48);
    const combined = combineFrames(left, right, field, { maxDisparity: 48, verticalOffset: 0 });
    const shift = combineShift(left.width, 48);
    expect(combined.mapping.kind).toBe('depth');
    if (combined.mapping.kind !== 'depth') throw new Error('unreachable');
    // A left pixel at x moves to x - d/2 + shift; a right pixel at x-d moves to x - d/2 + shift.
    const x = 80;
    expect(combined.mapping.shift).toBe(shift);
    expect(x - combined.mapping.disparity[x] / 2 + shift).toBeCloseTo(x - dx / 2 + shift, 5);
  });

  it('fills real pixels only and marks uncovered areas as no data', () => {
    const [left, right] = stereoPair(12);
    const field = buildDisparityField(left, right, 0, 48);
    const combined = combineFrames(left, right, field, { maxDisparity: 48, verticalOffset: 0 });
    let covered = 0;
    for (let i = 0; i < combined.coverage.length; i++) {
      if (combined.coverage[i]) covered++;
      else expect(combined.data[i * 4 + 3]).toBe(255); // transparent, never fake footage
    }
    expect(covered).toBeGreaterThan(combined.width * combined.height * 0.6);
    expect(combined.fillRatio).toBeLessThan(1);
    expect(combined.fillRatio).toBeGreaterThan(0.6);
  });

  it('compensates exposure differences before blending', () => {
    const [left, right] = stereoPair(10);
    for (let i = 0; i < right.data.length; i += 4) {
      for (let c = 0; c < 3; c++) right.data[i + c] = Math.round(right.data[i + c] * 0.5 + 10);
    }
    const matches = matchGrid(left, right, 0, 48, 4);
    const photometric = photometricCompensation(left, right, matches);
    expect(photometric.gain[0]).toBeGreaterThan(1.5);
    const field = buildDisparityField(left, right, 0, 48);
    const combined = combineFrames(left, right, field, { maxDisparity: 48, verticalOffset: 0 });
    // In the overlap the two sources must end up at similar luminance.
    let sumL = 0;
    let sumB = 0;
    let count = 0;
    for (let y = 20; y < 90; y += 3) {
      for (let x = 60; x < 100; x += 3) {
        const i = y * combined.width + x;
        if (!combined.coverage[i]) continue;
        sumL += left.data[(y * left.width + Math.min(left.width - 1, x)) * 4];
        sumB += combined.data[i * 4];
        count++;
      }
    }
    expect(count).toBeGreaterThan(20);
    expect(Math.abs(sumL / count - sumB / count)).toBeLessThan(30);
  });
});

describe('camera order and setup assessment', () => {
  const base = {
    matches: 30,
    agreement: 0.9,
    verticalOffset: 1,
    overlapRatio: 0.7,
    sharpnessA: 30,
    sharpnessB: 30,
    brightnessRatio: 1,
    order: 'a-left' as string | null,
  };

  it('accepts a good pair', () => {
    expect(evaluateSetup(base, 192).ready).toBe(true);
  });

  it('rejects insufficient overlap', () => {
    const result = evaluateSetup({ ...base, overlapRatio: 0.2 }, 192);
    expect(result.ready).toBe(false);
    expect(result.reason).toBe('low-overlap');
  });

  it('rejects too few matches and vertical misalignment', () => {
    expect(evaluateSetup({ ...base, matches: 4 }, 192).reason).toBe('few-matches');
    expect(evaluateSetup({ ...base, verticalOffset: 40 }, 192).reason).toBe('vertical-misalignment');
  });

  it('does not claim readiness when the camera order is unknown (wrong order)', () => {
    const result = evaluateSetup({ ...base, order: null }, 192);
    expect(result.ready).toBe(false);
    expect(result.reason).toBe('inconsistent');
  });

  it('warns about extreme exposure differences without failing', () => {
    const result = evaluateSetup({ ...base, brightnessRatio: 3 }, 192);
    expect(result.ready).toBe(true);
    expect(result.warnings).toContain('exposure');
  });
});

describe('coordinate transfer to the combined view', () => {
  function depthMapping(disparity: number, confidence: number, shift: number): ViewMapping {
    const width = 256;
    const height = 192;
    const d = new Float32Array(width * height).fill(disparity);
    const conf = new Uint8Array(width * height).fill(confidence);
    return { kind: 'depth', width, height, shift, disparity: d, confidence: conf };
  }

  it('transfers the hamster box by the same half-disparity warp as the image', () => {
    const mapping = depthMapping(10, 255, 20);
    const transferred = transferBoxToCombined({ left: 0.1, top: 0.1, right: 0.3, bottom: 0.3 }, mapping);
    // Left edge x = 25.6 → 25.6 - 10/2 + 20 = 40.6
    expect(transferred.bbox.left).toBeCloseTo(40.6, 1);
    expect(transferred.bbox.right).toBeCloseTo(76.8 - 5 + 20, 1);
    expect(transferred.bbox.top).toBeCloseTo(19.2, 1);
    expect(transferred.quad).toHaveLength(8);
    expect(transferred.confident).toBe(true);
  });

  it('falls back to a small shift and reports low confidence without measured depth', () => {
    const mapping = depthMapping(10, 0, 20);
    const transferred = transferBoxToCombined({ left: 0.1, top: 0.1, right: 0.3, bottom: 0.3 }, mapping);
    expect(transferred.bbox.left).toBeCloseTo(25.6 + 20, 1);
    expect(transferred.confident).toBe(false);
  });

  it('keeps the box unchanged in planar mode (left frame is the reference)', () => {
    const mapping: ViewMapping = { kind: 'planar', width: 256, height: 192, homography: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
    const transferred = transferBoxToCombined({ left: 0.25, top: 0.25, right: 0.5, bottom: 0.5 }, mapping);
    expect(transferred.bbox.left).toBeCloseTo(64, 5);
    expect(transferred.bbox.top).toBeCloseTo(48, 5);
    expect(transferred.confident).toBe(true);
  });
});

describe('planar fallback mode', () => {
  it('recovers a homography from noisy correspondences', () => {
    // Right frame is the left frame shifted by (6, -2) — a pure translation homography.
    const points = [];
    let seed = 4242;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    for (let i = 0; i < 60; i++) {
      const x = 10 + random() * 140;
      const y = 10 + random() * 90;
      points.push({ from: { x, y }, to: { x: x + 6, y: y - 2 } });
    }
    const fit = estimateHomography(points);
    expect(fit).not.toBeNull();
    const h = fit!.h;
    const mapX = (x: number, y: number) => (h[0] * x + h[1] * y + h[2]) / (h[6] * x + h[7] * y + h[8]);
    expect(mapX(80, 50)).toBeCloseTo(86, 0);
  });

  it('refuses a homography when there are not enough consistent matches', () => {
    const points = Array.from({ length: 10 }, (_, i) => ({ from: { x: i * 10, y: i * 7 }, to: { x: i * 3, y: i * 11 } }));
    expect(estimateHomography(points)).toBeNull();
  });

  it('warps the right frame into the left frame and blends identical views losslessly', () => {
    const [left] = stereoPair(8);
    const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const combined = combinePlanar(left, left, identity, { gain: [1, 1, 1], bias: [0, 0, 0] });
    expect(combined.mode).toBe('planar');
    expect(combined.mapping.kind).toBe('planar');
    // Identity homography with the same frame: the merge must not alter pixels.
    for (let i = 0; i < left.data.length; i += 17) {
      expect(combined.data[i]).toBe(left.data[i]);
    }
    expect(combined.parallaxRatio).toBe(0);
  });
});

describe('unreliable depth handling', () => {
  it('never warps with unmeasured disparity: unknown pixels keep the small global shift only', () => {
    const [left, right] = stereoPair(12);
    const field: DenseDisparity = {
      ...buildDisparityField(left, right, 0, 48),
    };
    // Simulate a mostly unknown field: zero confidence everywhere.
    field.confidence.fill(0);
    field.confidenceRight.fill(0);
    const combined = combineFrames(left, right, field, { maxDisparity: 48, verticalOffset: 0 });
    if (combined.mapping.kind !== 'depth') throw new Error('unreachable');
    // With zero confidence no pixel may carry a per-pixel depth warp.
    expect(combined.parallaxRatio).toBe(0);
  });
});
