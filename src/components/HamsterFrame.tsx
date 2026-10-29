import { useEffect, useRef } from 'react';
import type { HamsterObservation } from '../lib/hamster';

/** The box is drawn onto the analysed image, never onto a newer live frame. */
export default function HamsterFrame({ observation, className = '' }: { observation: HamsterObservation; className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    canvas.width = observation.width;
    canvas.height = observation.height;
    // ImageData needs its own ArrayBuffer in browsers with SharedArrayBuffer types.
    context.putImageData(new ImageData(new Uint8ClampedArray(observation.pixels), observation.width, observation.height), 0, 0);
    const box = observation.box;
    if (!box) return;
    const x = box.left * canvas.width;
    const y = box.top * canvas.height;
    const w = (box.right - box.left) * canvas.width;
    const h = (box.bottom - box.top) * canvas.height;
    const scale = canvas.width / 256;
    context.lineWidth = 2 * scale;
    context.strokeStyle = '#d6ffb7';
    context.shadowColor = '#12271a';
    context.shadowBlur = 3 * scale;
    context.strokeRect(x, y, w, h);
    context.shadowBlur = 0;
    const text = `Хомяк · ${Math.round(box.score * 100)}%`;
    context.font = `bold ${10 * scale}px sans-serif`;
    const labelWidth = context.measureText(text).width + 10 * scale;
    const labelX = Math.max(0, Math.min(canvas.width - labelWidth, x));
    const labelY = y >= 16 * scale ? y - 15 * scale : Math.min(canvas.height - 15 * scale, y + 2 * scale);
    context.fillStyle = '#1c3829';
    context.fillRect(labelX, labelY, labelWidth, 14 * scale);
    context.fillStyle = '#e2ffd0';
    context.fillText(text, labelX + 5 * scale, labelY + 10 * scale);
  }, [observation]);
  return <canvas ref={canvasRef} className={className} aria-label={observation.box ? 'Обработанный кадр с рамкой предполагаемого хомяка' : 'Обработанный кадр левой камеры'} />;
}
