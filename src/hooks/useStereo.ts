import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import type { HamsterObservation } from '../lib/hamster';
import type { CameraOrder, OrientationResult, PointCloudData, StereoSettings } from '../lib/stereo';
import type { WorkerRequest } from '../lib/stereo.worker';
import type { CameraSlots } from './useCameras';

export type OrderStatus = 'idle' | 'detecting' | 'automatic' | 'manual' | 'failed';
export type HamsterStatus = 'waiting' | 'loading' | 'ready' | 'error';

type WorkerResponse =
  | { kind: 'detect'; result: OrientationResult }
  | { kind: 'reconstruct'; result: PointCloudData; frameId: number; observation: HamsterObservation | null }
  | { kind: 'hamster-status'; status: 'ready' | 'error'; message?: string }
  | { kind: 'error'; message: string };

const WIDTH = 256;
const HEIGHT = 192;
const DETAIL_WIDTH = 512;
const DETAIL_HEIGHT = 384;

function frameFromVideo(video: HTMLVideoElement, detail: CanvasRenderingContext2D, stereo: CanvasRenderingContext2D, keepDetail: boolean) {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const targetRatio = WIDTH / HEIGHT;
  const sourceRatio = sourceWidth / sourceHeight;
  let sx = 0;
  let sy = 0;
  let sw = sourceWidth;
  let sh = sourceHeight;
  // Draw one central 4:3 snapshot; both resolutions come from that same image.
  if (sourceRatio > targetRatio) {
    sw = sourceHeight * targetRatio;
    sx = (sourceWidth - sw) / 2;
  } else {
    sh = sourceWidth / targetRatio;
    sy = (sourceHeight - sh) / 2;
  }
  detail.drawImage(video, sx, sy, sw, sh, 0, 0, DETAIL_WIDTH, DETAIL_HEIGHT);
  stereo.drawImage(detail.canvas, 0, 0, WIDTH, HEIGHT);
  return {
    stereo: stereo.getImageData(0, 0, WIDTH, HEIGHT).data,
    detail: keepDetail ? detail.getImageData(0, 0, DETAIL_WIDTH, DETAIL_HEIGHT).data : null,
  };
}

function median(values: number[]) {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered.length ? ordered[Math.floor(ordered.length / 2)] : 0;
}

interface Options {
  slots: CameraSlots;
  videoA: RefObject<HTMLVideoElement | null>;
  videoB: RefObject<HTMLVideoElement | null>;
  active: boolean;
  baselineCm: number;
  horizontalFov: number;
  maxDepthM: number;
}

/** Capture on the UI thread; feature matching and triangulation run in a worker. */
export function useStereo({ slots, videoA, videoB, active, baselineCm, horizontalFov, maxDepthM }: Options) {
  const [status, setStatus] = useState<OrderStatus>('idle');
  const [order, setOrder] = useState<CameraOrder | null>(null);
  const [verticalOffsetAB, setVerticalOffsetAB] = useState(0);
  const [orientation, setOrientation] = useState<OrientationResult | null>(null);
  const [cloud, setCloud] = useState<PointCloudData | null>(null);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [processingMs, setProcessingMs] = useState<number | null>(null);
  const [hamsterStatus, setHamsterStatus] = useState<HamsterStatus>('waiting');
  const [observation, setObservation] = useState<HamsterObservation | null>(null);
  const frameCounter = useRef(0);
  const [round, setRound] = useState(0);

  // Never keep an old detection on screen when capture has stopped or stalled.
  useEffect(() => {
    if (!observation) return;
    const remaining = Math.max(0, 5000 - (Date.now() - observation.capturedAt));
    const timer = window.setTimeout(() => {
      setObservation((current) => current?.frameId === observation.frameId ? null : current);
    }, remaining);
    return () => window.clearTimeout(timer);
  }, [observation]);

  useEffect(() => {
    setCloud(null);
    setOrder(null);
    setVerticalOffsetAB(0);
    setOrientation(null);
    setProcessingError(null);
    setProcessingMs(null);
    setObservation(null);
    setHamsterStatus('waiting');
    setStatus(slots.a && slots.b ? 'detecting' : 'idle');
  }, [slots.a?.stream, slots.b?.stream]);

  const detectAgain = useCallback(() => {
    if (!slots.a || !slots.b) return;
    setCloud(null);
    setObservation(null);
    setHamsterStatus('waiting');
    setOrder(null);
    setOrientation(null);
    setProcessingError(null);
    setStatus('detecting');
    setRound((value) => value + 1);
  }, [slots.a, slots.b]);

  const chooseOrder = useCallback((newOrder: CameraOrder) => {
    if (!slots.a || !slots.b) return;
    setCloud(null);
    setObservation(null);
    setProcessingError(null);
    setOrder(newOrder);
    setStatus('manual');
  }, [slots.a, slots.b]);

  const retryHamster = useCallback(() => {
    setObservation(null);
    setRound((value) => value + 1);
  }, []);

  useEffect(() => {
    setObservation(null);
    if (!active || !slots.a || !slots.b || (status !== 'detecting' && !order)) {
      setHamsterStatus('waiting');
      return;
    }
    const canvasA = document.createElement('canvas');
    const canvasB = document.createElement('canvas');
    const stereoA = document.createElement('canvas');
    const stereoB = document.createElement('canvas');
    canvasA.width = canvasB.width = DETAIL_WIDTH;
    canvasA.height = canvasB.height = DETAIL_HEIGHT;
    stereoA.width = stereoB.width = WIDTH;
    stereoA.height = stereoB.height = HEIGHT;
    const ctxA = canvasA.getContext('2d', { willReadFrequently: true });
    const ctxB = canvasB.getContext('2d', { willReadFrequently: true });
    const lowA = stereoA.getContext('2d', { willReadFrequently: true });
    const lowB = stereoB.getContext('2d', { willReadFrequently: true });
    if (!ctxA || !ctxB || !lowA || !lowB) {
      setProcessingError('Не удалось прочитать изображения камер.');
      if (status === 'detecting') setStatus('failed');
      return;
    }
    let worker: Worker;
    try {
      worker = new Worker(new URL('../lib/stereo.worker.ts', import.meta.url), { type: 'module' });
    } catch {
      setProcessingError('Не удалось запустить обработку стереопары. Проверьте настройки браузера.');
      setHamsterStatus('error');
      if (status === 'detecting') setStatus('failed');
      return;
    }
    if (order && status !== 'detecting') {
      setHamsterStatus('loading');
      const modelUrl = new URL(`${import.meta.env.BASE_URL}models/hamster-ssd-mobilenet-v2.onnx`, window.location.href).href;
      worker.postMessage({ kind: 'init-hamster', modelUrl } satisfies WorkerRequest);
    } else {
      setHamsterStatus('waiting');
    }
    let closed = false;
    let timer = 0;
    let attempts = 0;
    let sentAt = 0;
    const waitingSince = performance.now();
    const votes: OrientationResult[] = [];
    const measuredOffsets: number[] = [];

    const schedule = (delay: number) => {
      if (!closed) timer = window.setTimeout(capture, delay);
    };
    const capture = () => {
      if (closed) return;
      const a = videoA.current;
      const b = videoB.current;
      if (!a || !b || a.readyState < 2 || b.readyState < 2 || !a.videoWidth || !b.videoWidth) {
        if (performance.now() - waitingSince > 12000) {
          setProcessingError('Потоки камер не запустились. Проверьте доступ к устройствам и переподключите их.');
          if (status === 'detecting') setStatus('failed');
          return;
        }
        schedule(300);
        return;
      }
      try {
        const dataA = frameFromVideo(a, ctxA, lowA, status !== 'detecting' && order === 'a-left');
        const dataB = frameFromVideo(b, ctxB, lowB, status !== 'detecting' && order === 'b-left');
        const detail = dataA.detail ?? dataB.detail;
        const settings: StereoSettings = {
          baselineCm,
          horizontalFov,
          maxDepthM,
          verticalOffset: order === 'b-left' ? -verticalOffsetAB : verticalOffsetAB,
        };
        const request: WorkerRequest = {
          kind: status === 'detecting' ? 'detect' : 'reconstruct',
          width: WIDTH,
          height: HEIGHT,
          a: dataA.stereo.buffer as ArrayBuffer,
          b: dataB.stereo.buffer as ArrayBuffer,
          detail: detail ? { width: DETAIL_WIDTH, height: DETAIL_HEIGHT, data: detail.buffer as ArrayBuffer } : undefined,
          frameId: ++frameCounter.current,
          capturedAt: Date.now(),
          order: order ?? undefined,
          settings,
        };
        sentAt = performance.now();
        const transfer = [request.a, request.b];
        if (request.detail) transfer.push(request.detail.data);
        worker.postMessage(request, transfer);
      } catch {
        setProcessingError('Не удалось получить кадры. Переподключите камеры и повторите попытку.');
        if (status === 'detecting') setStatus('failed');
      }
    };

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      if (closed) return;
      const message = event.data;
      if (message.kind === 'hamster-status') {
        setHamsterStatus(message.status);
        if (message.status === 'error') setObservation(null);
        return;
      }
      if (message.kind === 'error') {
        setProcessingError('Не удалось обработать изображения камер. Проверьте настройки и повторите попытку.');
        setObservation(null);
        if (status === 'detecting') setStatus('failed');
        return;
      }
      setProcessingMs(Math.round(performance.now() - sentAt));
      if (message.kind === 'detect') {
        attempts++;
        const result = message.result;
        setOrientation(result);
        if (result.matches >= 4) measuredOffsets.push(result.verticalOffset);
        if (result.order) votes.push(result);
        const aVotes = votes.filter((vote) => vote.order === 'a-left');
        const bVotes = votes.filter((vote) => vote.order === 'b-left');
        const majority = aVotes.length >= bVotes.length ? aVotes : bVotes;
        if (majority.length >= 3 && majority.length / votes.length >= 0.75) {
          setVerticalOffsetAB(median(majority.map((vote) => vote.verticalOffset)));
          setOrder(majority[0].order);
          setStatus('automatic');
          return;
        }
        if (attempts >= 12) {
          setVerticalOffsetAB(median(measuredOffsets));
          setStatus('failed');
          return;
        }
        schedule(260);
      } else {
        setCloud(message.result);
        // Detection, image and point cloud were all produced from this same pair.
        const observed = message.observation;
        setObservation(observed && observed.frameId === message.frameId && Date.now() - observed.capturedAt < 5000 ? observed : null);
        schedule(450);
      }
    };
    worker.onerror = () => {
      if (closed) return;
      setProcessingError('Не удалось запустить обработку стереопары. Обновите страницу и попробуйте снова.');
      setHamsterStatus('error');
      setObservation(null);
      if (status === 'detecting') setStatus('failed');
    };
    schedule(100);
    return () => {
      closed = true;
      window.clearTimeout(timer);
      worker.terminate();
    };
  }, [active, slots.a?.stream, slots.b?.stream, videoA, videoB, status, order, verticalOffsetAB, baselineCm, horizontalFov, maxDepthM, round]);

  return { status, order, orientation, cloud, processingError, processingMs, detectAgain, chooseOrder, hamsterStatus, observation, retryHamster };
}
