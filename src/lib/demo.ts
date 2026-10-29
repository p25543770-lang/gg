import { depthColor, type PointCloudData } from './stereo';

type RGB = [number, number, number];
type Vector = [number, number, number];

/** An explicitly labelled, procedural sample; it contains no camera data. */
export function createDemoCloud(): PointCloudData {
  const positions: number[] = [];
  const rgbColors: number[] = [];
  const depthColors: number[] = [];
  let seed = 93812;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const add = (x: number, y: number, z: number, color: RGB) => {
    const jitter = 0.011;
    positions.push(x + (random() - 0.5) * jitter, y + (random() - 0.5) * jitter, z + (random() - 0.5) * jitter);
    const shade = 0.87 + random() * 0.22;
    rgbColors.push(...color.map((c) => Math.min(1, c * shade / 255)));
    const depth = depthColor(-z, 8);
    depthColors.push(depth[0] / 255 * shade, depth[1] / 255 * shade, depth[2] / 255 * shade);
  };
  const plane = (origin: Vector, u: Vector, v: Vector, stepsU: number, stepsV: number, color: RGB, density = 0.86) => {
    for (let i = 0; i <= stepsU; i++) {
      for (let j = 0; j <= stepsV; j++) {
        if (random() > density) continue;
        const a = i / stepsU;
        const b = j / stepsV;
        add(origin[0] + a * u[0] + b * v[0], origin[1] + a * u[1] + b * v[1], origin[2] + a * u[2] + b * v[2], color);
      }
    }
  };
  const box = (x: number, y: number, z: number, w: number, h: number, d: number, color: RGB, spacing = 0.065) => {
    const nx = Math.max(1, Math.round(w / spacing));
    const ny = Math.max(1, Math.round(h / spacing));
    const nz = Math.max(1, Math.round(d / spacing));
    plane([x, y + h, z], [w, 0, 0], [0, 0, d], nx, nz, color);
    plane([x, y, z + d], [w, 0, 0], [0, h, 0], nx, ny, color);
    plane([x + w, y, z], [0, 0, d], [0, h, 0], nz, ny, color, 0.75);
    plane([x, y, z], [0, 0, d], [0, h, 0], nz, ny, color, 0.7);
  };

  // Floor and the two visible room planes.
  plane([-3.2, -1.05, -6.2], [6.4, 0, 0], [0, 0, 5.8], 91, 82, [111, 163, 163], 0.75);
  plane([-3.2, -1.05, -6.2], [6.4, 0, 0], [0, 3.35, 0], 86, 44, [78, 123, 131], 0.62);
  plane([-3.2, -1.05, -6.2], [0, 0, 5.8], [0, 3.35, 0], 78, 44, [72, 113, 120], 0.38);

  // Window and its frame, slightly in front of the back wall.
  plane([-1.92, 0.42, -6.14], [2.18, 0, 0], [0, 1.45, 0], 36, 24, [103, 184, 190], 0.45);
  box(-2.0, 0.35, -6.1, 0.08, 1.6, 0.07, [180, 219, 208], 0.055);
  box(0.26, 0.35, -6.1, 0.08, 1.6, 0.07, [180, 219, 208], 0.055);
  box(-2.0, 1.87, -6.1, 2.34, 0.08, 0.07, [180, 219, 208], 0.055);
  box(-2.0, 0.35, -6.1, 2.34, 0.08, 0.07, [180, 219, 208], 0.055);
  box(-0.88, 0.35, -6.08, 0.07, 1.6, 0.07, [162, 205, 199], 0.055);

  // Table, slender legs and a monitor.
  box(0.3, 0.03, -4.5, 2.15, 0.12, 1.25, [184, 205, 180]);
  box(0.43, -1.05, -4.38, 0.1, 1.08, 0.1, [113, 168, 159]);
  box(2.2, -1.05, -4.38, 0.1, 1.08, 0.1, [113, 168, 159]);
  box(0.43, -1.05, -3.46, 0.1, 1.08, 0.1, [113, 168, 159]);
  box(2.2, -1.05, -3.46, 0.1, 1.08, 0.1, [113, 168, 159]);
  box(1.13, 0.15, -4.02, 0.24, 0.05, 0.28, [198, 221, 202], 0.04);
  box(1.22, 0.19, -3.95, 0.06, 0.23, 0.05, [198, 221, 202], 0.04);
  box(0.72, 0.42, -4.01, 1.1, 0.68, 0.08, [150, 216, 211], 0.045);
  plane([0.78, 0.48, -3.92], [0.98, 0, 0], [0, 0.56, 0], 25, 14, [91, 162, 166], 0.9);
  box(1.96, 0.15, -3.68, 0.19, 0.2, 0.19, [207, 223, 188], 0.04);

  // A low chair in the foreground.
  box(-1.52, -0.58, -3.54, 1.22, 0.27, 1.0, [184, 210, 192], 0.055);
  box(-1.52, -0.4, -3.55, 1.22, 0.94, 0.22, [171, 203, 192], 0.055);
  box(-1.55, -0.72, -3.52, 0.15, 0.52, 0.98, [180, 210, 196], 0.055);
  box(-0.45, -0.72, -3.52, 0.15, 0.52, 0.98, [180, 210, 196], 0.055);
  box(-1.43, -1.04, -3.42, 0.08, 0.47, 0.08, [128, 177, 165], 0.04);
  box(-0.48, -1.04, -3.42, 0.08, 0.47, 0.08, [128, 177, 165], 0.04);
  box(-1.43, -1.04, -2.68, 0.08, 0.47, 0.08, [128, 177, 165], 0.04);
  box(-0.48, -1.04, -2.68, 0.08, 0.47, 0.08, [128, 177, 165], 0.04);

  // Plant and pot: a few point-sampled leaves add a recognisable organic shape.
  box(-2.65, -1.05, -4.9, 0.55, 0.48, 0.55, [157, 192, 176], 0.05);
  for (let leaf = 0; leaf < 14; leaf++) {
    const angle = (leaf / 14) * Math.PI * 2;
    const reach = 0.45 + random() * 0.4;
    const rise = 0.65 + random() * 0.7;
    for (let t = 0; t < 34; t++) {
      const p = t / 33;
      const spread = Math.sin(p * Math.PI) * 0.13;
      for (let side = -2; side <= 2; side++) {
        if (random() > 0.8) continue;
        add(
          -2.38 + Math.cos(angle) * reach * p + Math.cos(angle + Math.PI / 2) * spread * side / 2,
          -0.58 + rise * p - 0.28 * p * p,
          -4.62 + Math.sin(angle) * reach * p + Math.sin(angle + Math.PI / 2) * spread * side / 2,
          [156, 220, 169],
        );
      }
    }
  }

  const width = 256;
  const height = 192;
  const depthMap = new Uint8ClampedArray(width * height * 4);
  const zBuffer = new Float32Array(width * height).fill(Infinity);
  for (let i = 0; i < width * height; i++) {
    depthMap[i * 4] = 13;
    depthMap[i * 4 + 1] = 31;
    depthMap[i * 4 + 2] = 36;
    depthMap[i * 4 + 3] = 255;
  }
  const focal = width / (2 * Math.tan(Math.PI / 6));
  for (let i = 0; i < positions.length; i += 3) {
    const z = -positions[i + 2];
    if (z <= 0) continue;
    const x = Math.round(width / 2 + positions[i] * focal / z);
    const y = Math.round(height / 2 - positions[i + 1] * focal / z);
    const color = depthColor(z, 8);
    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        const px = x + ox;
        const py = y + oy;
        if (px < 0 || py < 0 || px >= width || py >= height) continue;
        const p = py * width + px;
        if (z >= zBuffer[p]) continue;
        zBuffer[p] = z;
        depthMap[p * 4] = color[0];
        depthMap[p * 4 + 1] = color[1];
        depthMap[p * 4 + 2] = color[2];
      }
    }
  }
  return {
    positions: new Float32Array(positions),
    rgbColors: new Float32Array(rgbColors),
    depthColors: new Float32Array(depthColors),
    depthMap,
    // The procedural demo has no corresponding real left-camera pixels.
    pixelIndices: new Uint32Array(0),
    width,
    height,
    count: positions.length / 3,
    near: null,
    far: null,
  };
}
