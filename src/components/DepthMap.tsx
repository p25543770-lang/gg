import { useEffect, useRef } from 'react';
import type { PointCloudData } from '../lib/stereo';

export default function DepthMap({ cloud }: { cloud: PointCloudData | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    if (!cloud) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    canvas.width = cloud.width;
    canvas.height = cloud.height;
    const image = context.createImageData(cloud.width, cloud.height);
    image.data.set(cloud.depthMap);
    context.putImageData(image, 0, 0);
  }, [cloud]);
  return <canvas ref={ref} className="depth-map-canvas" aria-label="Приблизительная карта глубины" />;
}
