import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import type { HamsterObservation } from '../lib/hamster';
import type { CameraOrder, OrientationResult, PointCloudData, StereoSettings } from '../lib/stereo';
import type { CombinedPayload, CombineStatus as WorkerCombineStatus, WorkerRequest, WorkerResponse } from '../lib/stereo.worker';
import type { MappedQuad, SetupAssessment } from '../lib/stitch';
import { evaluateSetup } from '../lib/stitch';
import type { CameraSlots } from './useCameras';

export type OrderStatus = 'idle' | 'detecting' | 'automatic' | 'manual' | 'failed';
export type HamsterStatus = 'waiting' | 'loading' | 'ready' | 'error';
export type CombineStatus =
  | 'none'
  | 'setup'
  | 'ready'
  | 'planar'
  | 'unreliable-depth'
  | 'low-overlap'
  | 'insufficient'
  | 'desync'
  | 'error';

export interface PairMetrics {
  matches: number;
  overlapRatio: number;
  verticalOffset: number;
  disparityMedian: number;
  disparitySpread: number;
  brightnessRatio: number;
  agreement: number;
}

export interface PerfStats {
  /** Displayed combined frames per second (EMA). */
  fps: number;
  /** Capture → processed result, milliseconds. */
  latencyMs: number | null;
  /** Worker processing time of the last frame, milliseconds. */
  processingMs: number | null;
  /** Source timestamp difference between the two streams, milliseconds. */
  skewMs: number;
  /** Processing resolution after automatic quality scaling. */
  workWidth: number;
  workHeight: number;
}

export interface SetupProfile {
  aId: string;
  bId: string;
  order: CameraOrder;
  verticalOffset: number;
  sourceWidth: number;
  sourceHeight: number;
  baselineCm: number;
  savedAt: number;
}

const PROFILE_KEY = 'stereo-space-profile-v1';
const SKEW_LIMIT_MS = 150;
const COMBINE_MAX_AGE_MS = 2500;

function loadProfile(): SetupProfile | null {
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as SetupProfile;
    if (parsed && typeof parsed.order === 'string' && typeof parsed.aId === 'string' && typeof parsed.bId === 'string') return parsed;
    return null;
  } catch {
    return null;
  }
}

function saveProfile(profile: SetupProfile) {
  try {
    localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  } catch {
    // Storage is optional; the app works without it.
  }
}

function median(values: number[]) {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered.length ? ordered[Math.floor(ordered.length / 2)] : 0;
}

function frameFromVideo(video: HTMLVideoElement, detail: CanvasRenderingContext2D, stereo: CanvasRenderingContext2D, keepDetail: boolean, width: number, height: number, detailWidth: number, detailHeight: number) {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const targetRatio = width / height;
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
  detail.drawImage(video, sx, sy, sw, sh, 0, 0, detailWidth, detailHeight);
  stereo.drawImage(detail.canvas, 0, 0, width, height);
  return {
    stereo: stereo.getImageData(0, 0, width, height).data,
    detail: keepDetail ? detail.getImageData(0, 0, detailWidth, detailHeight).data : null,
  };
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

/** Capture on the UI thread; matching, merging and triangulation run in a worker. */
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
  const [combined, setCombined] = useState<CombinedPayload | null>(null);
  const [boxOnCombined, setBoxOnCombined] = useState<MappedQuad | null>(null);
  const [combineStatus, setCombineStatus] = useState<CombineStatus>('none');
  const [pairMetrics, setPairMetrics] = useState<PairMetrics | null>(null);
  const [setupAssessment, setSetupAssessment] = useState<SetupAssessment | null>(null);
  const [perf, setPerf] = useState<PerfStats>({ fps: 0, latencyMs: null, processingMs: null, skewMs: 0, workWidth: 256, workHeight: 192 });
  const [profileNote, setProfileNote] = useState<'loaded' | 'saved' | 'stale' | null>(null);
  const frameCounter = useRef(0);
  const [round, setRound] = useState(0);
  const profileRef = useRef<SetupProfile | null>(loadProfile());
  const qualityRef = useRef(0);
  const slowFramesRef = useRef(0);
  const lastShownRef = useRef(0);
  const processingEmaRef = useRef(0);

  // Never keep an old detection on screen when capture has stopped or stalled.
  useEffect(() => {
    if (!observation) return;
    const remaining = Math.max(0, 5000 - (Date.now() - observation.capturedAt));
    const timer = window.setTimeout(() => {
      setObservation((current) => current?.frameId === observation.frameId ? null : current);
    }, remaining);
    return () => window.clearTimeout(timer);
  }, [observation]);

  // Never present a stale combined frame as the current one.
  const capturedAtRef = useRef(0);

  useEffect(() => {
    setCloud(null);
    setOrder(null);
    setVerticalOffsetAB(0);
    setOrientation(null);
    setProcessingError(null);
    setProcessingMs(null);
    setObservation(null);
    setHamsterStatus('waiting');
    setCombined(null);
    setBoxOnCombined(null);
    setPairMetrics(null);
    setSetupAssessment(null);
    setPerf({ fps: 0, latencyMs: null, processingMs: null, skewMs: 0, workWidth: 256, workHeight: 192 });
    qualityRef.current = 0;
    slowFramesRef.current = 0;
    processingEmaRef.current = 0;

    const aId = slots.a?.id ?? '';
    const bId = slots.b?.id ?? '';
    const video = videoA.current;
    const sourceWidth = video?.videoWidth ?? 0;
    const sourceHeight = video?.videoHeight ?? 0;
    const profile = profileRef.current;
    if (slots.a && slots.b && profile) {
      const sameDevices = profile.aId === aId && profile.bId === bId;
      const sameMode = !sourceWidth || !profile.sourceWidth || (profile.sourceWidth === sourceWidth && profile.sourceHeight === sourceHeight);
      if (sameDevices && sameMode) {
        // Known pair and mode: restore measured geometry, skip re-detection.
        setOrder(profile.order);
        setVerticalOffsetAB(profile.verticalOffset);
        setStatus('automatic');
        setCombineStatus('setup');
        setProfileNote('loaded');
        return;
      }
      // The devices or the capture mode changed: the old profile cannot be trusted.
      setProfileNote('stale');
      profileRef.current = null;
      try { localStorage.removeItem(PROFILE_KEY); } catch { /* optional */ }
    } else {
      setProfileNote(null);
    }
    setCombineStatus(slots.a && slots.b ? 'setup' : 'none');
    setStatus(slots.a && slots.b ? 'detecting' : 'idle');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slots.a?.stream, slots.b?.stream]);

  const detectAgain = useCallback(() => {
    if (!slots.a || !slots.b) return;
    setCloud(null);
    setCombined(null);
    setBoxOnCombined(null);
    setObservation(null);
    setHamsterStatus('waiting');
    setOrder(null);
    setOrientation(null);
    setSetupAssessment(null);
    setPairMetrics(null);
    setProcessingError(null);
    setCombineStatus('setup');
    setStatus('detecting');
    setRound((value) => value + 1);
  }, [slots.a, slots.b]);

  const chooseOrder = useCallback((newOrder: CameraOrder) => {
    if (!slots.a || !slots.b) return;
    setCloud(null);
    setCombined(null);
    setBoxOnCombined(null);
    setObservation(null);
    setProcessingError(null);
    setOrder(newOrder);
    setStatus('manual');
    setCombineStatus('setup');
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
    const quality = qualityRef.current;
    const WIDTH = quality === 0 ? 256 : quality === 1 ? 192 : 160;
    const HEIGHT = quality === 0 ? 192 : quality === 1 ? 144 : 120;
    const DETAIL_WIDTH = 512;
    const DETAIL_HEIGHT = 384;
    setPerf((current) => ({ ...current, workWidth: WIDTH, workHeight: HEIGHT }));
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
    let captureAt = 0;
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
      // Software pairing by presentation time — this is NOT hardware sync.
      const timeA = Number.isFinite(a.currentTime) ? a.currentTime : 0;
      const timeB = Number.isFinite(b.currentTime) ? b.currentTime : 0;
      const skewMs = Math.abs(timeA - timeB) * 1000;
      setPerf((current) => ({ ...current, skewMs }));
      if (status !== 'detecting' && skewMs > SKEW_LIMIT_MS) {
        // Frames too far apart in time: merging would double moving objects.
        setCombined(null);
        setBoxOnCombined(null);
        setCombineStatus('desync');
        schedule(300);
        return;
      }
      try {
        captureAt = performance.now();
        capturedAtRef.current = Date.now();
        const dataA = frameFromVideo(a, ctxA, lowA, status !== 'detecting' && order === 'a-left', WIDTH, HEIGHT, DETAIL_WIDTH, DETAIL_HEIGHT);
        const dataB = frameFromVideo(b, ctxB, lowB, status !== 'detecting' && order === 'b-left', WIDTH, HEIGHT, DETAIL_WIDTH, DETAIL_HEIGHT);
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
        setCombineStatus('error');
        if (status === 'detecting') setStatus('failed');
        return;
      }
      const now = performance.now();
      const duration = Math.round(now - sentAt);
      processingEmaRef.current = processingEmaRef.current ? processingEmaRef.current * 0.7 + duration * 0.3 : duration;
      if (processingEmaRef.current > 800) {
        slowFramesRef.current++;
        if (slowFramesRef.current >= 3 && qualityRef.current < 2) {
          qualityRef.current++;
          slowFramesRef.current = 0;
          setRound((value) => value + 1);
        }
      } else {
        slowFramesRef.current = 0;
      }
      const latencyMs = Math.round(now - captureAt);
      setProcessingMs(duration);
      setPerf((current) => ({
        ...current,
        latencyMs,
        processingMs: duration,
        fps: current.fps ? current.fps * 0.7 + (1000 / Math.max(1, now - (lastShownRef.current || now))) * 0.3 : 0,
      }));
      lastShownRef.current = now;

      if (message.kind === 'detect') {
        attempts++;
        const result = message.result;
        setOrientation(result);
        setPairMetrics({
          matches: result.matches,
          overlapRatio: result.overlapRatio ?? 0,
          verticalOffset: result.verticalOffset,
          disparityMedian: result.medianDisparity,
          disparitySpread: result.disparitySpread ?? 0,
          brightnessRatio: result.brightnessRatio ?? 1,
          agreement: result.agreement,
        });
        if (result.matches >= 4) measuredOffsets.push(result.verticalOffset);
        if (result.order) votes.push(result);
        const aVotes = votes.filter((vote) => vote.order === 'a-left');
        const bVotes = votes.filter((vote) => vote.order === 'b-left');
        const majority = aVotes.length >= bVotes.length ? aVotes : bVotes;
        if (majority.length >= 3 && majority.length / votes.length >= 0.75) {
          // verticalOffset is kept in the y(A)-y(B) convention everywhere.
          const offset = median(majority.map((vote) => vote.verticalOffset));
          const chosen = majority[0].order;
          if (!chosen) return;
          setVerticalOffsetAB(offset);
          setOrder(chosen);
          setStatus('automatic');
          const assessment = evaluateSetup({ ...result, order: chosen, verticalOffset: offset }, HEIGHT);
          setSetupAssessment(assessment);
          if (assessment.ready) {
            const video = videoA.current;
            const profile: SetupProfile = {
              aId: slots.a?.id ?? '',
              bId: slots.b?.id ?? '',
              order: chosen,
              verticalOffset: offset,
              sourceWidth: video?.videoWidth ?? 0,
              sourceHeight: video?.videoHeight ?? 0,
              baselineCm,
              savedAt: Date.now(),
            };
            profileRef.current = profile;
            saveProfile(profile);
            setProfileNote('saved');
          } else {
            setCombineStatus('low-overlap');
          }
          return;
        }
        if (attempts >= 12) {
          const offset = median(measuredOffsets);
          setVerticalOffsetAB(offset);
          const reason = result.overlapRatio < 0.35 ? 'low-overlap' : 'few-matches';
          setSetupAssessment({ ready: false, reason, warnings: [] });
          setStatus('failed');
          setCombineStatus(reason === 'low-overlap' ? 'low-overlap' : 'insufficient');
          return;
        }
        schedule(260);
      } else {
        setCloud(message.result);
        // Detection, image and point cloud were all produced from this same pair.
        const observed = message.observation;
        setObservation(observed && observed.frameId === message.frameId && Date.now() - observed.capturedAt < 5000 ? observed : null);
        const age = Date.now() - capturedAtRef.current;
        const workerStatus: WorkerCombineStatus = message.combineStatus ?? 'insufficient';
        const payload = message.combined ?? null;
        if (payload && age < COMBINE_MAX_AGE_MS) {
          setCombined(payload);
          setBoxOnCombined(message.boxOnCombined ?? null);
        } else {
          setCombined(null);
          setBoxOnCombined(null);
        }
        if (!payload) {
          setCombineStatus(workerStatus === 'ready' ? 'insufficient' : workerStatus);
        } else {
          setCombineStatus(workerStatus);
        }
        schedule(450);
      }
    };
    worker.onerror = () => {
      if (closed) return;
      setProcessingError('Не удалось запустить обработку стереопары. Обновите страницу и попробуйте снова.');
      setHamsterStatus('error');
      setObservation(null);
      setCombineStatus('error');
      if (status === 'detecting') setStatus('failed');
    };
    schedule(100);
    return () => {
      closed = true;
      window.clearTimeout(timer);
      worker.terminate();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, slots.a?.stream, slots.b?.stream, videoA, videoB, status, order, verticalOffsetAB, baselineCm, horizontalFov, maxDepthM, round]);

  return {
    status,
    order,
    orientation,
    cloud,
    processingError,
    processingMs,
    detectAgain,
    chooseOrder,
    hamsterStatus,
    observation,
    retryHamster,
    combined,
    boxOnCombined,
    combineStatus,
    pairMetrics,
    setupAssessment,
    perf,
    profileNote,
  };
}
