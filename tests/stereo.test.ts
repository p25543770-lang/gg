import { describe, expect, it } from 'vitest';
import { depthColor, inferCameraOrder, reconstructStereo, type Frame } from '../src/lib/stereo';

function stereoPair(dx: number, dy = 0, initialSeed = 12345): [Frame, Frame] {
  const width = 160;
  const height = 112;
  const a = new Uint8ClampedArray(width * height * 4);
  const b = new Uint8ClampedArray(a.length);
  let seed = initialSeed;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed >>> 24;
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
  // x(A)-x(B)=dx, y(A)-y(B)=dy.
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

describe('stereo ordering', () => {
  it('finds the left camera from positive disparity and the vertical offset', () => {
    const [a, b] = stereoPair(12, 3);
    const result = inferCameraOrder(a, b);
    expect(result.order).toBe('a-left');
    expect(result.verticalOffset).toBe(3);
    expect(result.medianDisparity).toBe(12);
    expect(result.matches).toBeGreaterThan(8);
  });

  it('recognises the opposite input order', () => {
    const [left, right] = stereoPair(9, -2);
    const result = inferCameraOrder(right, left);
    expect(result.order).toBe('b-left');
    expect(result.verticalOffset).toBe(2);
    expect(result.medianDisparity).toBe(-9);
  });

  it('does not guess an order for featureless images or zero disparity', () => {
    const [a] = stereoPair(0);
    const blank = { ...a, data: new Uint8ClampedArray(a.data.length).fill(128) };
    expect(inferCameraOrder(blank, blank).order).toBeNull();
    expect(inferCameraOrder(a, a).order).toBeNull();
  });

  it('does not infer an order from unrelated scenes', () => {
    const [a] = stereoPair(8, 0, 12345);
    const [b] = stereoPair(8, 0, 987654);
    expect(inferCameraOrder(a, b).order).toBeNull();
  });

  it('tolerates a different camera exposure', () => {
    const [a, b] = stereoPair(10);
    for (let i = 0; i < b.data.length; i += 4) {
      for (let c = 0; c < 3; c++) b.data[i + c] = Math.round(b.data[i + c] * 0.8 + 20);
    }
    expect(inferCameraOrder(a, b).order).toBe('a-left');
  });
});

describe('stereo reconstruction', () => {
  it('triangulates positive disparity with the specified baseline and field of view', () => {
    const [left, right] = stereoPair(12, 2);
    const result = reconstructStereo(left, right, {
      baselineCm: 6,
      horizontalFov: 60,
      maxDepthM: 8,
      verticalOffset: 2,
    });
    const expectedDepth = (160 / (2 * Math.tan(Math.PI / 6))) * 0.06 / 12;
    expect(result.count).toBeGreaterThan(500);
    expect(result.near).toBeCloseTo(expectedDepth, 3);
    expect(result.far).toBeCloseTo(expectedDepth, 3);
    expect(result.positions[2]).toBeCloseTo(-expectedDepth, 3);
    expect(result.depthMap).toHaveLength(160 * 112 * 4);
    expect(result.pixelIndices).toHaveLength(result.count);
    const focal = 160 / (2 * Math.tan(Math.PI / 6));
    const index = result.pixelIndices[0];
    const depth = -result.positions[2];
    expect(result.positions[0]).toBeCloseTo(((index % 160) - 80) * depth / focal, 5);
    expect(result.positions[1]).toBeCloseTo((56 - Math.floor(index / 160)) * depth / focal, 5);
    const widerBaseline = reconstructStereo(left, right, {
      baselineCm: 12, horizontalFov: 60, maxDepthM: 8, verticalOffset: 2,
    });
    expect(widerBaseline.near).toBeCloseTo(expectedDepth * 2, 3);
  });

  it('does not fabricate points for flat images', () => {
    const [a] = stereoPair(0);
    const blank = { ...a, data: new Uint8ClampedArray(a.data.length).fill(100) };
    const result = reconstructStereo(blank, blank, {
      baselineCm: 6.5,
      horizontalFov: 60,
      maxDepthM: 8,
      verticalOffset: 0,
    });
    expect(result.count).toBe(0);
    expect(result.near).toBeNull();
  });

  it('does not assign positive depth to identical textured frames', () => {
    const [a] = stereoPair(0);
    const result = reconstructStereo(a, a, { baselineCm: 6.5, horizontalFov: 60, maxDepthM: 8, verticalOffset: 0 });
    expect(result.count).toBe(0);
  });

  it('rejects invalid parameters', () => {
    const [a, b] = stereoPair(8);
    expect(() => reconstructStereo(a, b, { baselineCm: 0, horizontalFov: 60, maxDepthM: 8, verticalOffset: 0 })).toThrow();
    expect(depthColor(0.2, 8)).not.toEqual(depthColor(8, 8));
  });
});
