import { useEffect, useRef, type RefObject } from 'react';
import { drawCornerMarks, type Corner, type DrawStyle } from '../lib/corners';

interface Props {
  /** Corners in processed-frame coordinates (central 4:3 crop pixels). */
  corners: Corner[];
  /** The live video element the marks are drawn over. */
  videoRef: RefObject<HTMLVideoElement | null>;
  /** How the video is fitted into its box: cover crops, contain letterboxes. */
  fit: 'cover' | 'contain';
  style?: DrawStyle;
}

/**
 * Purple stroke marks exactly on the corners the worker detected. The marks
 * are mapped from the processed 4:3 crop back through the frame crop and the
 * video's display fit, so they sit on the real image features.
 */
export default function CornerOverlay({ corners, videoRef, fit, style }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    const video = videoRef.current;
    if (!canvas || !context) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.clearRect(0, 0, width, height);
    const videoWidth = video?.videoWidth ?? 0;
    const videoHeight = video?.videoHeight ?? 0;
    if (!videoWidth || !videoHeight) return;

    // Displayed content box of the fitted video inside the overlay.
    const scale = fit === 'cover'
      ? Math.max(width / videoWidth, height / videoHeight)
      : Math.min(width / videoWidth, height / videoHeight);
    const offsetX = (width - videoWidth * scale) / 2;
    const offsetY = (height - videoHeight * scale) / 2;

    // The processed frame is the central 4:3 crop of the source stream —
    // the same crop frameFromVideo draws.
    const targetRatio = 4 / 3;
    const sourceRatio = videoWidth / videoHeight;
    let cropX = 0;
    let cropY = 0;
    let cropWidth = videoWidth;
    let cropHeight = videoHeight;
    if (sourceRatio > targetRatio) {
      cropWidth = videoHeight * targetRatio;
      cropX = (videoWidth - cropWidth) / 2;
    } else {
      cropHeight = videoWidth / targetRatio;
      cropY = (videoHeight - cropHeight) / 2;
    }

    const points = corners.map((corner) => {
      const sourceX = cropX + (corner.x / 256) * cropWidth;
      const sourceY = cropY + (corner.y / 192) * cropHeight;
      return { x: offsetX + sourceX * scale, y: offsetY + sourceY * scale };
    });
    drawCornerMarks(context, points, { size: 4, ...style });
  }, [corners, videoRef, fit, style]);

  return <canvas ref={canvasRef} className="corner-overlay" aria-label="Найденные углы, отмеченные фиолетовыми метками" />;
}
