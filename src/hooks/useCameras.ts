import { useCallback, useEffect, useRef, useState } from 'react';

export interface CameraOption {
  id: string;
  label: string;
}

export interface CameraSlot extends CameraOption {
  stream: MediaStream;
}

export interface CameraSlots {
  a: CameraSlot | null;
  b: CameraSlot | null;
}

const EMPTY_SLOTS: CameraSlots = { a: null, b: null };

function stopStream(stream: MediaStream | undefined) {
  stream?.getTracks().forEach((track) => track.stop());
}

function explainError(error: unknown): string {
  const name = typeof error === 'object' && error !== null && 'name' in error ? String(error.name) : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Нет доступа к камерам. Разрешите его в браузере. Если доступ заблокирован в окне предпросмотра, откройте страницу отдельно.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError' || name === 'DevicesNotFoundError') {
    return 'Камера не найдена. Проверьте подключение и обновите список устройств.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError' || name === 'AbortError') {
    return 'Не удалось запустить камеру. Она может быть занята другим приложением, либо браузер не поддерживает одновременную работу двух камер.';
  }
  return 'Не удалось подключить камеры. Проверьте разрешения браузера и попробуйте ещё раз.';
}

function constraints(deviceId?: string): MediaStreamConstraints {
  return {
    audio: false,
    video: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      width: { ideal: 640 },
      height: { ideal: 480 },
      aspectRatio: { ideal: 4 / 3 },
    },
  };
}

/** Owns every MediaStream it opens, including partial and cancelled requests. */
export function useCameras() {
  const [slots, setSlots] = useState<CameraSlots>(EMPTY_SLOTS);
  const [devices, setDevices] = useState<CameraOption[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const slotsRef = useRef<CameraSlots>(EMPTY_SLOTS);
  const operationRef = useRef(0);
  const mountedRef = useRef(false);
  const permissionRef = useRef(false);

  const publish = useCallback((slot: 'a' | 'b', camera: CameraSlot | null) => {
    const next = { ...slotsRef.current, [slot]: camera };
    slotsRef.current = next;
    if (mountedRef.current) setSlots(next);
    if (camera) {
      const videoTrack = camera.stream.getVideoTracks()[0];
      videoTrack?.addEventListener('ended', () => {
        if (!mountedRef.current || slotsRef.current[slot]?.stream !== camera.stream) return;
        publish(slot, null);
        setError('Одна из камер отключилась. Проверьте устройство и подключите пару заново.');
      }, { once: true });
    }
  }, []);

  const listDevices = useCallback(async (): Promise<CameraOption[]> => {
    const all = await navigator.mediaDevices.enumerateDevices();
    const seen = new Set<string>();
    const cameras = all
      .filter((device) => device.kind === 'videoinput' && device.deviceId && !seen.has(device.deviceId) && seen.add(device.deviceId))
      .map((device, index) => ({ id: device.deviceId, label: device.label || `Камера ${index + 1}` }));
    if (mountedRef.current) setDevices(cameras);
    return cameras;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    const onDeviceChange = () => {
      if (permissionRef.current) void listDevices().catch(() => {});
    };
    navigator.mediaDevices?.addEventListener?.('devicechange', onDeviceChange);
    return () => {
      mountedRef.current = false;
      operationRef.current++;
      navigator.mediaDevices?.removeEventListener?.('devicechange', onDeviceChange);
      stopStream(slotsRef.current.a?.stream);
      stopStream(slotsRef.current.b?.stream);
      slotsRef.current = EMPTY_SLOTS;
    };
  }, [listDevices]);

  const disconnect = useCallback(() => {
    operationRef.current++;
    stopStream(slotsRef.current.a?.stream);
    stopStream(slotsRef.current.b?.stream);
    slotsRef.current = EMPTY_SLOTS;
    setSlots(EMPTY_SLOTS);
    setBusy(false);
    setError(null);
  }, []);

  const connect = useCallback(async () => {
    const token = ++operationRef.current;
    stopStream(slotsRef.current.a?.stream);
    stopStream(slotsRef.current.b?.stream);
    slotsRef.current = EMPTY_SLOTS;
    setSlots(EMPTY_SLOTS);
    setError(null);
    setBusy(true);
    if (!navigator.mediaDevices?.getUserMedia || !navigator.mediaDevices?.enumerateDevices) {
      setError('Камеры недоступны. Откройте страницу через HTTPS или localhost и используйте современный браузер.');
      setBusy(false);
      return;
    }
    let initial: MediaStream | undefined;
    let second: MediaStream | undefined;
    try {
      initial = await navigator.mediaDevices.getUserMedia(constraints());
      if (!mountedRef.current || operationRef.current !== token) { stopStream(initial); return; }
      permissionRef.current = true;
      const available = await listDevices();
      if (!mountedRef.current || operationRef.current !== token) { stopStream(initial); return; }
      const firstTrack = initial.getVideoTracks()[0];
      if (!firstTrack || firstTrack.readyState !== 'live') throw new DOMException('Camera not ready', 'NotReadableError');
      const firstId = firstTrack.getSettings().deviceId || available[0]?.id || '';
      const firstLabel = available.find((device) => device.id === firstId)?.label || firstTrack.label || 'Камера 1';
      publish('a', { id: firstId, label: firstLabel, stream: initial });
      // The published stream is now owned by slotsRef, including on errors below.
      initial = undefined;
      const other = available.find((device) => device.id !== firstId);
      if (!other) {
        setError('Найдена только одна камера. Подключите вторую, обновите список и выберите её в карточке камеры 02.');
        return;
      }
      second = await navigator.mediaDevices.getUserMedia(constraints(other.id));
      if (!mountedRef.current || operationRef.current !== token) { stopStream(second); return; }
      const secondTrack = second.getVideoTracks()[0];
      if (!secondTrack || secondTrack.readyState !== 'live' || firstTrack.readyState !== 'live') {
        throw new DOMException('Cameras cannot run together', 'NotReadableError');
      }
      const secondId = secondTrack.getSettings().deviceId || other.id;
      if (firstId && secondId === firstId) {
        throw new DOMException('Identical camera selected twice', 'NotReadableError');
      }
      publish('b', { id: secondId, label: available.find((device) => device.id === secondId)?.label || secondTrack.label || other.label, stream: second });
      second = undefined;
    } catch (caught) {
      stopStream(initial);
      stopStream(second);
      if (mountedRef.current && operationRef.current === token) {
        // Some browsers stop the first stream when a second camera is opened.
        const firstCamera = slotsRef.current.a;
        if (firstCamera && firstCamera.stream.getVideoTracks()[0]?.readyState !== 'live') {
          stopStream(firstCamera.stream);
          publish('a', null);
        }
        setError(explainError(caught));
      }
    } finally {
      if (mountedRef.current && operationRef.current === token) setBusy(false);
    }
  }, [listDevices, publish]);

  const changeCamera = useCallback(async (slot: 'a' | 'b', id: string) => {
    if (!id || !navigator.mediaDevices?.getUserMedia || slotsRef.current[slot]?.id === id) return;
    const otherSlot = slot === 'a' ? 'b' : 'a';
    if (slotsRef.current[otherSlot]?.id === id) {
      setError('Выберите две разные камеры.');
      return;
    }
    const token = ++operationRef.current;
    setBusy(true);
    setError(null);
    let stream: MediaStream | undefined;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints(id));
      if (!mountedRef.current || operationRef.current !== token) { stopStream(stream); return; }
      const track = stream.getVideoTracks()[0];
      const actualId = track?.getSettings().deviceId || id;
      if (!track || track.readyState !== 'live' || actualId === slotsRef.current[otherSlot]?.id) {
        throw new DOMException('Camera not available', 'NotReadableError');
      }
      stopStream(slotsRef.current[slot]?.stream);
      const label = devices.find((device) => device.id === actualId)?.label || track.label || 'Камера';
      publish(slot, { id: actualId, label, stream });
      stream = undefined;
    } catch (caught) {
      stopStream(stream);
      if (mountedRef.current && token === operationRef.current) setError(explainError(caught));
    } finally {
      if (mountedRef.current && token === operationRef.current) setBusy(false);
    }
  }, [devices, publish]);

  const refresh = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const available = await listDevices();
      if (available.length >= 2) setError(null);
      else if (permissionRef.current) setError('Найдена только одна камера. Подключите вторую и снова обновите список.');
    } catch (caught) {
      setError(explainError(caught));
    }
  }, [listDevices]);

  return { slots, devices, busy, error, connect, disconnect, changeCamera, refresh, clearError: () => setError(null) };
}
