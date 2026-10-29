import { useEffect, useRef } from 'react';
import { drawCornerMarks } from '../lib/corners';
import type { MappedQuad } from '../lib/stitch';
import type { CombinedPayload } from '../lib/stereo.worker';

interface Props {
  combined: CombinedPayload;
  box: MappedQuad | null;
  showBox: boolean;
  /** Left-camera corners already warped into combined-view pixels. */
  corners?: Array<{ x: number; y: number }> | null;
  showCorners?: boolean;
}

/**
 * Renders exactly the pixels the two cameras really captured. Areas outside
 * both fields of view stay empty and are hatched — they are never filled with
 * invented content. The hamster box is drawn as the quad produced by the same
 * geometric warp the image went through.
 */
export default function CombinedView({ combined, box, showBox, corners, showCorners = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    canvas.width = combined.width;
    canvas.height = combined.height;

    // Empty background first: dark hatch marks "no camera data".
    context.fillStyle = '#101f24';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.save();
    context.strokeStyle = 'rgba(120, 150, 155, 0.18)';
    context.lineWidth = 1;
    for (let x = -canvas.height; x < canvas.width; x += 8) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x + canvas.height, canvas.height);
      context.stroke();
    }
    context.restore();

    const image = context.createImageData(combined.width, combined.height);
    const pixels = new Uint8ClampedArray(combined.data);
    const coverage = new Uint8ClampedArray(combined.coverage);
    for (let i = 0; i < coverage.length; i++) {
      if (!coverage[i]) {
        // Fully transparent: the hatch shows through, clearly not footage.
        pixels[i * 4 + 3] = 0;
      }
    }
    image.data.set(pixels);
    context.putImageData(image, 0, 0);

    const scale = canvas.width / 320;
    if (showBox && box) {
      context.lineWidth = 2 * scale;
      context.strokeStyle = box.confident ? '#d6ffb7' : '#ffd28a';
      context.shadowColor = '#12271a';
      context.shadowBlur = 3 * scale;
      context.beginPath();
      box.quad.slice(0, 4).forEach((point, index) => {
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.stroke();
      context.shadowBlur = 0;
      const text = box.confident ? 'Хомяк' : 'Хомяк · положение приблизительное';
      context.font = `bold ${10 * scale}px sans-serif`;
      const labelWidth = context.measureText(text).width + 10 * scale;
      const labelX = Math.max(0, Math.min(canvas.width - labelWidth, box.bbox.left));
      const labelY = box.bbox.top >= 16 * scale ? box.bbox.top - 15 * scale : Math.min(canvas.height - 15 * scale, box.bbox.bottom + 2 * scale);
      context.fillStyle = box.confident ? '#1c3829' : '#3c2c12';
      context.fillRect(labelX, labelY, labelWidth, 14 * scale);
      context.fillStyle = box.confident ? '#e2ffd0' : '#ffe6bd';
      context.fillText(text, labelX + 5 * scale, labelY + 10 * scale);
    }

    if (showCorners && corners && corners.length) {
      // Purple stroke marks on the corners, transferred by the same warp.
      drawCornerMarks(context, corners, { size: 4 * scale, lineWidth: 1.6 * scale });
    }
  }, [combined, box, showBox, corners, showCorners]);

  return (
    <canvas
      ref={canvasRef}
      className="combined-canvas"
      aria-label="Совмещённый видеопоток двух камер"
    />
  );
}
