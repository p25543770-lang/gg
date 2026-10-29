import { describe, expect, it } from 'vitest';
import { describeHamsterPosition, estimateHamsterDepth, extractRegion, hamsterSearchRegions, mapRegionBox, prepareHamsterInput, selectHamster, type HamsterBox } from '../src/lib/hamster';
import type { PointCloudData } from '../src/lib/stereo';

const box: HamsterBox = { left: 0.1, top: 0.1, right: 0.9, bottom: 0.9, score: 0.7 };

function cloudFromSamples(samples: [number, number, number][]): PointCloudData {
  const positions: number[] = [];
  const pixelIndices: number[] = [];
  for (const [x, y, z] of samples) {
    positions.push(0, 0, -z);
    pixelIndices.push(y * 100 + x);
  }
  return {
    width: 100, height: 100, count: samples.length,
    positions: new Float32Array(positions), pixelIndices: new Uint32Array(pixelIndices),
    rgbColors: new Float32Array(0), depthColors: new Float32Array(0), depthMap: new Uint8ClampedArray(0),
    near: null, far: null,
  };
}

const spread: [number, number, number][] = [
  [30, 30, 1], [35, 35, 1], [65, 30, 1], [70, 35, 1],
  [30, 65, 1], [35, 70, 1], [65, 65, 1], [70, 70, 1],
];

describe('Hamster model preprocessing and output', () => {
  it('resizes RGB like the original TF1 bilinear operation and ignores alpha', () => {
    const pixels = new Uint8ClampedArray([
      0, 10, 20, 0,     100, 110, 120, 255,
      200, 210, 220, 1,  240, 250, 255, 123,
    ]);
    const input = prepareHamsterInput(pixels, 2, 2);
    expect(input).toHaveLength(300 * 300 * 3);
    expect(Array.from(input.slice(0, 3))).toEqual([0, 10, 20]);
    // At pixel (75, 75), the source coordinate is exactly (0.5, 0.5).
    expect(Array.from(input.slice((75 * 300 + 75) * 3, (75 * 300 + 75) * 3 + 3))).toEqual([135, 145, 153.75]);
    expect(Array.from(input.slice(-3))).toEqual([240, 250, 255]);
    expect(() => prepareHamsterInput(pixels, 3, 2)).toThrow();
  });

  it('uses the best valid hamster score, normalises and clips the box', () => {
    const boxes = [0.1, 0.2, 0.8, 0.9, -0.2, 0.3, 1.2, 1.1, 0.5, 0.5, 0.5, 0.6];
    const scores = [0.3, 0.7, 0.99];
    expect(selectHamster(boxes, scores)).toEqual({ left: 0.3, top: 0, right: 1, bottom: 1, score: 0.7 });
  });

  it('does not invent detections below threshold or from invalid data', () => {
    expect(selectHamster([0, 0, 1, 1], [0.249])).toBeNull();
    expect(selectHamster([0, NaN, 1, 1], [0.8])).toBeNull();
    expect(selectHamster([0, 0, 1, 1], [1.2])).toBeNull();
    expect(() => selectHamster([0, 1], [0.9])).toThrow();
  });

  it('extracts overlapping search regions and maps complete boxes to the full frame', () => {
    const regions = hamsterSearchRegions(256, 192);
    expect(regions).toHaveLength(9);
    expect(regions[0]).toEqual({ x: 64, y: 48, width: 128, height: 96 });
    const pixels = new Uint8ClampedArray(256 * 192 * 4);
    pixels[(48 * 256 + 64) * 4] = 123;
    const centre = extractRegion(pixels, 256, 192, regions[0]);
    expect(centre).toHaveLength(128 * 96 * 4);
    expect(centre[0]).toBe(123);
    const mapped = mapRegionBox({ left: 0.2, right: 0.8, top: 0.2, bottom: 0.8, score: 0.4 }, regions[0], 256, 192)!;
    expect(mapped.left).toBeCloseTo(0.35);
    expect(mapped.right).toBeCloseTo(0.65);
    expect(mapped.top).toBeCloseTo(0.35);
    expect(mapped.bottom).toBeCloseTo(0.65);
    expect(mapped.score).toBe(0.4);
    expect(mapRegionBox({ ...box, left: 0 }, regions[0], 256, 192)).toBeNull();
    expect(() => extractRegion(pixels, 256, 192, { x: 250, y: 0, width: 20, height: 20 })).toThrow();
  });

  it('describes the centre of the analysed crop, not the whole camera sensor', () => {
    expect(describeHamsterPosition(box)).toBe('по центру, посередине по высоте');
    expect(describeHamsterPosition({ ...box, left: 0.01, right: 0.2, top: 0.01, bottom: 0.2 })).toBe('слева, сверху');
    expect(describeHamsterPosition({ ...box, left: 0.8, right: 1, top: 0.8, bottom: 1 })).toBe('справа, снизу');
  });
});

describe('Conservative hamster depth from the same left frame', () => {
  it('uses the median of enough spatially spread, consistent samples', () => {
    const estimate = estimateHamsterDepth(box, cloudFromSamples([...spread, [1, 1, 50]]));
    expect(estimate).toEqual({ metres: 1, samples: 8 });
  });

  it('does not measure when there are too few points, even with a detected box', () => {
    expect(estimateHamsterDepth(box, cloudFromSamples(spread.slice(0, 7)))).toEqual({ metres: null, samples: 7 });
  });

  it('rejects points concentrated in one part of the box', () => {
    const clustered: [number, number, number][] = Array.from({ length: 12 }, (_, i) => [29 + i % 3, 29 + Math.floor(i / 3), 1]);
    expect(estimateHamsterDepth(box, cloudFromSamples(clustered)).metres).toBeNull();
  });

  it('rejects conflicting depths rather than averaging foreground and background', () => {
    const mixed: [number, number, number][] = spread.map(([x, y], i) => [x, y, i < 4 ? 1 : 3]);
    expect(estimateHamsterDepth(box, cloudFromSamples(mixed))).toEqual({ metres: null, samples: 8 });
  });

  it('ignores invalid depths and samples outside the inner part of the box', () => {
    const samples: [number, number, number][] = [[10, 10, 2], [30, 30, NaN], [31, 31, 0], [32, 32, Infinity]];
    expect(estimateHamsterDepth(box, cloudFromSamples(samples))).toEqual({ metres: null, samples: 0 });
  });
});
