import { describe, expect, it } from 'vitest';
import type { Frame } from '../src/lib/stereo';
import { CORNER_COLOR, detectCorners, drawCornerMarks } from '../src/lib/corners';
import { transferPointToCombined, type ViewMapping } from '../src/lib/stitch';

/** Alternating squares with mild noise: classic corner-detection target. */
function checkerboard(square = 16, width = 128, height = 96, noise = 6): Frame {
  const data = new Uint8ClampedArray(width * height * 4);
  let seed = 20261029;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed >>> 24) % (2 * noise + 1) - noise;
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const light = (Math.floor(x / square) + Math.floor(y / square)) % 2 === 0;
      const value = Math.max(0, Math.min(255, (light ? 215 : 40) + random()));
      const p = (y * width + x) * 4;
      data[p] = value;
      data[p + 1] = value;
      data[p + 2] = value;
      data[p + 3] = 255;
    }
  }
  return { width, height, data };
}

function flat(width = 128, height = 96, value = 128): Frame {
  const data = new Uint8ClampedArray(width * height * 4).fill(value);
  for (let i = 3; i < data.length; i += 4) data[i] = 255;
  return { width, height, data };
}

describe('corner detection (Shi-Tomasi)', () => {
  it('finds the intersections of a checkerboard and marks exactly those points', () => {
    const square = 16;
    const frame = checkerboard(square);
    const corners = detectCorners(frame, { maxCorners: 40, minDistance: 8 });
    expect(corners.length).toBeGreaterThanOrEqual(8);
    // True corner positions: multiples of the square size inside the margins.
    const truth: Array<{ x: number; y: number }> = [];
    for (let y = square; y < frame.height - 2; y += square) {
      for (let x = square; x < frame.width - 2; x += square) truth.push({ x, y });
    }
    for (const corner of corners) {
      const nearest = Math.min(...truth.map((t) => Math.hypot(t.x - corner.x, t.y - corner.y)));
      expect(nearest).toBeLessThan(3);
    }
  });

  it('finds no corners on a flat frame', () => {
    expect(detectCorners(flat())).toHaveLength(0);
  });

  it('keeps marks spaced apart and never exceeds the cap', () => {
    const corners = detectCorners(checkerboard(8, 128, 96, 10), { maxCorners: 12, minDistance: 9 });
    expect(corners.length).toBeLessThanOrEqual(12);
    for (let i = 0; i < corners.length; i++) {
      for (let j = i + 1; j < corners.length; j++) {
        expect(Math.hypot(corners[i].x - corners[j].x, corners[i].y - corners[j].y)).toBeGreaterThanOrEqual(9);
      }
    }
  });

  it('rejects malformed frames instead of crashing', () => {
    expect(detectCorners({ width: 8, height: 8, data: new Uint8ClampedArray(8) })).toHaveLength(0);
  });
});

describe('purple corner marks', () => {
  it('draws two stroke lines per corner in purple', () => {
    const strokes: Array<{ from: number[]; to: number[] }> = [];
    let strokeStyle = '';
    const context = {
      save() {},
      restore() {},
      beginPath() {},
      stroke() {},
      moveTo(x: number, y: number) { this.current = [x, y]; },
      lineTo(x: number, y: number) { strokes.push({ from: (this as unknown as { current: number[] }).current, to: [x, y] }); },
      set strokeStyle(value: string) { strokeStyle = value; },
      get strokeStyle() { return strokeStyle; },
      lineWidth: 0,
      lineCap: '',
      shadowColor: '',
      shadowBlur: 0,
      current: [0, 0],
    } as unknown as CanvasRenderingContext2D & { current: number[] };
    drawCornerMarks(context, [{ x: 10, y: 20 }, { x: 40, y: 50 }], { size: 4 });
    // Two strokes (horizontal + vertical) per corner.
    expect(strokes).toHaveLength(4);
    expect(strokeStyle).toBe(CORNER_COLOR);
    expect(strokeStyle).toMatch(/purple|#[0-9a-f]{6}/i);
    expect(strokes[0].from).toEqual([6, 20]);
    expect(strokes[0].to).toEqual([14, 20]);
    expect(strokes[1].from).toEqual([10, 16]);
    expect(strokes[1].to).toEqual([10, 24]);
  });

  it('transfers corner marks to the combined view through the image warp', () => {
    const width = 256;
    const height = 192;
    const mapping: ViewMapping = {
      kind: 'depth',
      width,
      height,
      shift: 20,
      disparity: new Float32Array(width * height).fill(10),
      confidence: new Uint8Array(width * height).fill(255),
    };
    const moved = transferPointToCombined({ x: 100, y: 50 }, mapping);
    expect(moved.x).toBeCloseTo(100 - 5 + 20, 5);
    expect(moved.y).toBe(50);
  });
});
