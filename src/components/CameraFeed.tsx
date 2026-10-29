import { useEffect, type RefObject } from 'react';
import { Camera, ChevronDown } from 'lucide-react';
import type { CameraSlot, CameraOption } from '../hooks/useCameras';
import type { HamsterObservation } from '../lib/hamster';
import type { Corner } from '../lib/corners';
import CornerOverlay from './CornerOverlay';
import HamsterFrame from './HamsterFrame';

interface Props {
  slot: 'a' | 'b';
  camera: CameraSlot | null;
  otherId?: string;
  devices: CameraOption[];
  videoRef: RefObject<HTMLVideoElement | null>;
  position: string;
  busy: boolean;
  observation: HamsterObservation | null;
  corners: Corner[];
  cornersOn: boolean;
  onChange: (slot: 'a' | 'b', id: string) => void;
}

export default function CameraFeed({ slot, camera, otherId, devices, videoRef, position, busy, observation, corners, cornersOn, onChange }: Props) {
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = camera?.stream ?? null;
    if (camera) void video.play().catch(() => { /* Browser will retry with autoplay. */ });
    return () => { video.srcObject = null; };
  }, [camera, videoRef]);

  const index = slot === 'a' ? '01' : '02';
  return (
    <div className="camera-card">
      <div className="camera-preview">
        <video ref={videoRef} autoPlay muted playsInline className={camera ? 'visible' : ''} />
        {camera && cornersOn && !observation?.box && <CornerOverlay corners={corners} videoRef={videoRef} fit="cover" />}
        {camera && observation?.box && <HamsterFrame observation={observation} className="camera-analysis-frame" />}
        {!camera && (
          <div className="camera-empty">
            <Camera size={19} strokeWidth={1.4} />
            <span>Нет сигнала</span>
          </div>
        )}
        <span className="camera-number">CAM {index}</span>
        {camera && <span className="camera-live"><span /> {observation?.box ? 'КАДР' : 'LIVE'}</span>}
      </div>
      <div className="camera-card-info">
        <div className="camera-label-row">
          <span className="camera-name">Камера {index}</span>
          <span className={`camera-position ${position === 'Не определена' ? 'unknown' : ''}`}>{position}</span>
        </div>
        <div className="select-wrap">
          <select
            aria-label={`Устройство для камеры ${index}`}
            value={camera?.id ?? ''}
            disabled={busy || devices.length === 0}
            onChange={(event) => onChange(slot, event.target.value)}
            title={camera?.label ?? 'Выбрать камеру'}
          >
            {!camera && <option value="">Не подключена</option>}
            {camera && !devices.some((device) => device.id === camera.id) && <option value={camera.id}>{camera.label}</option>}
            {devices.map((device) => <option key={device.id} value={device.id} disabled={device.id === otherId}>{device.label}</option>)}
          </select>
          <ChevronDown size={13} aria-hidden="true" />
        </div>
      </div>
    </div>
  );
}
