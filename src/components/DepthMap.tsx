import { useEffect, useRef } from 'react';
import type { PointCloudData } from '../lib/stereo';

export default function DepthMap({ cloud, mode = 'depth' }: { cloud: PointCloudData | null; mode?: 'depth' | 'disparity' }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    const source = mode === 'disparity' ? cloud?.disparityMap ?? cloud?.depthMap : cloud?.depthMap;
    if (!cloud || !source) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    canvas.width = cloud.width;
    canvas.height = cloud.height;
    const image = context.createImageData(cloud.width, cloud.height);
    image.data.set(source);
    context.putImageData(image, 0, 0);
  }, [cloud, mode]);
  return <canvas ref={ref} className="depth-map-canvas" aria-label={mode === 'disparity' ? 'Карта диспарити в пикселях' : 'Приблизительная карта глубины'} />;
}
