import type { PointCloudData } from './stereo';

/** Export coordinates in metres, with +Y up and the cameras looking towards -Z. */
export function downloadPly(cloud: PointCloudData, useDepthColor: boolean) {
  const colors = useDepthColor ? cloud.depthColors : cloud.rgbColors;
  const header = [
    'ply',
    'format ascii 1.0',
    'comment Coordinates in metres; the stereo cameras look toward negative Z.',
    `element vertex ${cloud.count}`,
    'property float x',
    'property float y',
    'property float z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    'end_header',
  ];
  const lines = new Array<string>(cloud.count);
  for (let i = 0; i < cloud.count; i++) {
    const p = i * 3;
    const channel = (value: number) => Math.round(Math.min(1, Math.max(0, value)) * 255);
    lines[i] = `${cloud.positions[p].toFixed(5)} ${cloud.positions[p + 1].toFixed(5)} ${cloud.positions[p + 2].toFixed(5)} ${channel(colors[p])} ${channel(colors[p + 1])} ${channel(colors[p + 2])}`;
  }
  const blob = new Blob([[...header, ...lines].join('\n'), '\n'], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `stereo-space-${new Date().toISOString().slice(0, 10)}.ply`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
