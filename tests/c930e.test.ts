import { describe, expect, it } from 'vitest';
import {
  C930E,
  C930E_CROP_FOV_DEG,
  c930eConstraints,
  centralCropFovDeg,
  describeGeometry,
  depthFromDisparity,
  disparityFromDepth,
  focalPixels,
  isC930e,
  pickStereoDevices,
} from '../src/lib/c930e';
import { disparityColor, reconstructStereo, type Frame } from '../src/lib/stereo';

describe('Logitech C930e profile', () => {
  it('recognises C930e device labels', () => {
    expect(isC930e('Logitech C930e')).toBe(true);
    expect(isC930e('HD Pro Webcam C930e')).toBe(true);
    expect(isC930e('c930e')).toBe(true);
    expect(isC930e('Logitech HD Pro Webcam C920')).toBe(false);
    expect(isC930e(undefined)).toBe(false);
    expect(isC930e('')).toBe(false);
  });

  it('uses the official 90° diagonal FOV (82.1° horizontal, 52.2° vertical at 16:9)', () => {
    expect(C930E.diagonalFovDeg).toBe(90);
    expect(C930E.horizontalFovDeg).toBeCloseTo(82.1, 1);
    expect(C930E.verticalFovDeg).toBeCloseTo(52.2, 1);
  });

  it('narrows the FOV for the central 4:3 processing crop', () => {
    // The crop keeps the sensor height: hFOV_crop = 2*atan(tan(vFOV/2)*4/3).
    expect(C930E_CROP_FOV_DEG).toBeGreaterThan(65);
    expect(C930E_CROP_FOV_DEG).toBeLessThan(67.5);
    // A 16:9 "crop" is the full frame: unchanged horizontal FOV.
    expect(centralCropFovDeg(82.1, 52.2, 16 / 9)).toBeCloseTo(82.1, 0);
  });

  it('requests two-camera-friendly capture constraints', () => {
    const constraints = c930eConstraints('dev-1') as { audio: unknown; video: MediaTrackConstraints & { deviceId?: { exact?: string } } };
    expect(constraints.audio).toBe(false);
    expect(constraints.video.deviceId?.exact).toBe('dev-1');
    expect((constraints.video.width as ConstrainULongRange).ideal).toBe(1280);
    expect((constraints.video.height as ConstrainULongRange).ideal).toBe(720);
    expect((constraints.video.frameRate as ConstrainDoubleRange).ideal).toBe(30);
    expect(constraints.video.aspectRatio).toEqual({ ideal: 16 / 9 });
  });
});

describe('stereo depth formula Z = f * B / d', () => {
  it('computes focal length in pixels from the field of view', () => {
    // f = (W/2) / tan(hFOV/2); for the C930e crop at 256 px this is ~195.7.
    const f = focalPixels(256, C930E_CROP_FOV_DEG);
    expect(f).toBeGreaterThan(190);
    expect(f).toBeLessThan(200);
    // Focal scales linearly with resolution.
    expect(focalPixels(512, C930E_CROP_FOV_DEG)).toBeCloseTo(f * 2, 5);
    expect(() => focalPixels(0, 60)).toThrow();
  });

  it('measures depth as f * B / d and refuses unmeasurable disparity', () => {
    const f = focalPixels(256, C930E_CROP_FOV_DEG);
    const baselineM = 0.065;
    expect(depthFromDisparity(f, baselineM, 12)).toBeCloseTo((f * baselineM) / 12, 10);
    expect(depthFromDisparity(f, baselineM, 0)).toBeNull();
    expect(depthFromDisparity(f, baselineM, -3)).toBeNull();
    expect(depthFromDisparity(0, baselineM, 12)).toBeNull();
  });

  it('round-trips disparity ↔ depth', () => {
    const f = focalPixels(256, C930E_CROP_FOV_DEG);
    const depth = depthFromDisparity(f, 0.1, 8)!;
    expect(disparityFromDepth(f, 0.1, depth)).toBeCloseTo(8, 8);
    // Nearer objects need a larger pixel shift — the inverse proportionality.
    const near = depthFromDisparity(f, 0.1, 20)!;
    const far = depthFromDisparity(f, 0.1, 5)!;
    expect(near).toBeLessThan(far);
    expect(far / near).toBeCloseTo(4, 5);
  });

  it('describes the live geometry for the UI readout', () => {
    const geometry = describeGeometry(256, C930E_CROP_FOV_DEG, 6.5);
    expect(geometry.baselineM).toBeCloseTo(0.065, 5);
    expect(geometry.focalPx).toBeCloseTo(focalPixels(256, C930E_CROP_FOV_DEG), 5);
  });
});

describe('stereo pair picking', () => {
  it('prefers two Logitech C930e cameras', () => {
    const pair = pickStereoDevices([
      { id: '1', label: 'Logitech HD Pro Webcam C920' },
      { id: '2', label: 'HD Pro Webcam C930e' },
      { id: '3', label: 'Logitech C930e' },
    ]);
    expect(pair?.first.id).toBe('2');
    expect(pair?.second?.id).toBe('3');
    expect(pair?.bothC930e).toBe(true);
  });

  it('falls back to the first two different cameras and reports the pair state', () => {
    const pair = pickStereoDevices([
      { id: '1', label: 'Integrated Camera' },
      { id: '2', label: 'USB Camera' },
    ]);
    expect(pair?.first.id).toBe('1');
    expect(pair?.second?.id).toBe('2');
    expect(pair?.bothC930e).toBe(false);
  });

  it('handles a single camera and duplicate device ids', () => {
    expect(pickStereoDevices([{ id: '1', label: 'A' }])?.second).toBeNull();
    expect(pickStereoDevices([])).toBeNull();
    const pair = pickStereoDevices([{ id: '1', label: 'A' }, { id: '1', label: 'A copy' }, { id: '2', label: 'B' }]);
    expect(pair?.second?.id).toBe('2');
  });
});

describe('disparity map (OpenCV-style preview)', () => {
  function pair(dx: number): [Frame, Frame] {
    const width = 96;
    const height = 64;
    const a = new Uint8ClampedArray(width * height * 4);
    const b = new Uint8ClampedArray(a.length);
    let seed = 4242;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return (seed >>> 24) % 200 + 30;
    };
    for (let i = 0; i < width * height; i++) {
      const v = random();
      a[i * 4] = v; a[i * 4 + 1] = v; a[i * 4 + 2] = v; a[i * 4 + 3] = 255;
      b[i * 4] = random(); b[i * 4 + 1] = b[i * 4]; b[i * 4 + 2] = b[i * 4]; b[i * 4 + 3] = 255;
    }
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const bx = x - dx;
        if (bx < 0 || bx >= width) continue;
        b.set(a.subarray((y * width + x) * 4, (y * width + x) * 4 + 4), (y * width + bx) * 4);
      }
    }
    return [{ width, height, data: a }, { width, height, data: b }];
  }

  it('colours disparity blue for small shifts and red for large ones', () => {
    const small = disparityColor(2, 48);
    const large = disparityColor(48, 48);
    expect(small[2]).toBeGreaterThan(small[0]);
    expect(large[0]).toBeGreaterThan(large[2]);
  });

  it('reconstructStereo produces a disparity map alongside the depth map', () => {
    const [left, right] = pair(9);
    const result = reconstructStereo(left, right, {
      baselineCm: 6.5,
      horizontalFov: 66,
      maxDepthM: 8,
      verticalOffset: 0,
    });
    expect(result.disparityMap).toBeDefined();
    expect(result.disparityMap!.length).toBe(96 * 64 * 4);
    expect(result.count).toBeGreaterThan(100);
  });
});
