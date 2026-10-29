// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useStereo, type SetupProfile } from '../src/hooks/useStereo';
import type { CameraSlots } from '../src/hooks/useCameras';
import type { HamsterObservation } from '../src/lib/hamster';
import type { PointCloudData } from '../src/lib/stereo';

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  messages: Array<Record<string, unknown>> = [];
  terminated = false;
  constructor() { FakeWorker.instances.push(this); }
  postMessage(message: Record<string, unknown>) { this.messages.push(message); }
  terminate() { this.terminated = true; }
  emit(message: unknown) { this.onmessage?.({ data: message } as MessageEvent); }
}

const cloud: PointCloudData = {
  width: 256, height: 192, count: 0, near: null, far: null,
  positions: new Float32Array(0), pixelIndices: new Uint32Array(0),
  rgbColors: new Float32Array(0), depthColors: new Float32Array(0), depthMap: new Uint8ClampedArray(0),
};
function obs(frameId: number, capturedAt = Date.now()): HamsterObservation {
  return { frameId, capturedAt, leftSlot: 'a', width: 256, height: 192, pixels: new Uint8ClampedArray(256 * 192 * 4), box: null, distance: null };
}
const combinedPayload = {
  width: 320, height: 192,
  data: new ArrayBuffer(320 * 192 * 4),
  coverage: new ArrayBuffer(320 * 192),
  mode: 'depth' as const,
  fillRatio: 0.9,
  parallaxRatio: 0.8,
};
const boxOnCombined = {
  quad: Array.from({ length: 8 }, () => ({ x: 10, y: 10 })),
  bbox: { left: 5, top: 5, right: 15, bottom: 15 },
  confident: true,
};
const detectResult = {
  order: 'a-left' as const,
  matches: 30,
  agreement: 0.95,
  verticalOffset: 2,
  medianDisparity: 9,
  overlapRatio: 0.7,
  disparitySpread: 1,
  sharpnessA: 30,
  sharpnessB: 30,
  brightnessRatio: 1,
};

function makeSlots(aId = 'a', bId = 'b'): CameraSlots {
  return {
    a: { id: aId, label: 'A', stream: { id: `stream-${aId}` } as unknown as MediaStream },
    b: { id: bId, label: 'B', stream: { id: `stream-${bId}` } as unknown as MediaStream },
  };
}

function makeVideo(currentTime = 0) {
  return { current: { videoWidth: 640, videoHeight: 480, readyState: 4, currentTime } as unknown as HTMLVideoElement };
}

const options = (slots: CameraSlots, videoA = makeVideo(), videoB = makeVideo()) => ({
  slots, videoA, videoB, active: true, baselineCm: 6.5, horizontalFov: 60, maxDepthM: 8,
});

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  FakeWorker.instances = [];
  vi.stubGlobal('Worker', FakeWorker);
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation((function (this: HTMLCanvasElement) {
    return {
      canvas: this,
      drawImage: vi.fn(),
      getImageData: (_x: number, _y: number, width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
    };
  }) as typeof HTMLCanvasElement.prototype.getContext);
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function detectSession() {
  const slots = makeSlots();
  const hook = renderHook((props: ReturnType<typeof options>) => useStereo(props), { initialProps: options(slots) });
  act(() => vi.advanceTimersByTime(100));
  const worker = FakeWorker.instances.at(-1)!;
  expect(worker.messages[0].kind).toBe('detect');
  return { ...hook, worker, slots };
}

function manualSession() {
  const hook = detectSession();
  act(() => hook.result.current.chooseOrder('a-left'));
  const worker = FakeWorker.instances.at(-1)!;
  expect(worker.messages[0].kind).toBe('init-hamster');
  act(() => { worker.emit({ kind: 'hamster-status', status: 'ready' }); vi.advanceTimersByTime(100); });
  return { ...hook, worker };
}

describe('combined view lifecycle', () => {
  it('publishes the combined frame with its transferred box only from the same response', () => {
    const { result, worker, request } = (() => {
      const session = manualSession();
      const request = session.worker.messages.find((m) => m.kind === 'reconstruct')!;
      return { ...session, request };
    })();
    act(() => worker.emit({
      kind: 'reconstruct',
      frameId: request.frameId,
      result: cloud,
      observation: obs(request.frameId as number),
      combined: combinedPayload,
      boxOnCombined,
      combineStatus: 'ready',
    }));
    expect(result.current.combined).toBe(combinedPayload);
    expect(result.current.boxOnCombined).toBe(boxOnCombined);
    expect(result.current.combineStatus).toBe('ready');
  });

  it('never shows a stale combined frame as current', () => {
    const { result, worker } = manualSession();
    const request = worker.messages.find((m) => m.kind === 'reconstruct')!;
    // The capture is old by the time the result arrives.
    vi.setSystemTime(Date.now() + 5000);
    act(() => worker.emit({
      kind: 'reconstruct',
      frameId: request.frameId,
      result: cloud,
      observation: null,
      combined: combinedPayload,
      boxOnCombined,
      combineStatus: 'ready',
    }));
    expect(result.current.combined).toBeNull();
  });

  it('falls back to raw streams without a false ready status when merging fails', () => {
    const { result, worker } = manualSession();
    const request = worker.messages.find((m) => m.kind === 'reconstruct')!;
    act(() => worker.emit({
      kind: 'reconstruct',
      frameId: request.frameId,
      result: cloud,
      observation: null,
      combined: null,
      boxOnCombined: null,
      combineStatus: 'insufficient',
    }));
    expect(result.current.combined).toBeNull();
    expect(result.current.combineStatus).toBe('insufficient');
    expect(result.current.combineStatus).not.toBe('ready');
  });

  it('reports unreliable depth separately while still showing the merge', () => {
    const { result, worker } = manualSession();
    const request = worker.messages.find((m) => m.kind === 'reconstruct')!;
    act(() => worker.emit({
      kind: 'reconstruct',
      frameId: request.frameId,
      result: cloud,
      observation: null,
      combined: combinedPayload,
      boxOnCombined: null,
      combineStatus: 'unreliable-depth',
    }));
    expect(result.current.combineStatus).toBe('unreliable-depth');
    expect(result.current.combined).not.toBeNull();
  });
});

describe('frame pairing in time', () => {
  it('stops merging when the two streams are too far apart in time', () => {
    const slots = makeSlots();
    const videoA = makeVideo(1.0);
    const videoB = makeVideo(1.4); // 400 ms apart — beyond the software pairing limit
    const hook = renderHook((props: ReturnType<typeof options>) => useStereo(props), { initialProps: options(slots, videoA, videoB) });
    act(() => hook.result.current.chooseOrder('a-left'));
    act(() => vi.advanceTimersByTime(100));
    const worker = FakeWorker.instances.at(-1)!;
    const reconstructs = worker.messages.filter((m) => m.kind === 'reconstruct');
    expect(reconstructs).toHaveLength(0);
    expect(hook.result.current.combineStatus).toBe('desync');
  });

  it('accepts nearly simultaneous frames and sends them for processing', () => {
    const slots = makeSlots();
    const videoA = makeVideo(1.0);
    const videoB = makeVideo(1.02); // 20 ms apart — ordinary software pairing
    const hook = renderHook((props: ReturnType<typeof options>) => useStereo(props), { initialProps: options(slots, videoA, videoB) });
    act(() => hook.result.current.chooseOrder('a-left'));
    act(() => vi.advanceTimersByTime(100));
    const worker = FakeWorker.instances.at(-1)!;
    const reconstructs = worker.messages.filter((m) => m.kind === 'reconstruct');
    expect(reconstructs.length).toBe(1);
    expect(hook.result.current.combineStatus).not.toBe('desync');
  });
});

describe('camera order', () => {
  it('negates the measured vertical offset when the order is swapped', () => {
    const { result, worker } = detectSession();
    for (let i = 0; i < 3; i++) {
      act(() => worker.emit({ kind: 'detect', result: { ...detectResult, order: 'b-left', verticalOffset: 2 } }));
    }
    expect(result.current.order).toBe('b-left');
    const nextWorker = FakeWorker.instances.at(-1)!;
    act(() => vi.advanceTimersByTime(100));
    const request = nextWorker.messages.find((m) => m.kind === 'reconstruct')!;
    expect((request.settings as { verticalOffset: number }).verticalOffset).toBe(-2);
    expect(request.order).toBe('b-left');
  });

  it('keeps the offset for the a-left order', () => {
    const { result, worker } = detectSession();
    for (let i = 0; i < 3; i++) {
      act(() => worker.emit({ kind: 'detect', result: detectResult }));
    }
    expect(result.current.order).toBe('a-left');
    const nextWorker = FakeWorker.instances.at(-1)!;
    act(() => vi.advanceTimersByTime(100));
    const request = nextWorker.messages.find((m) => m.kind === 'reconstruct')!;
    expect((request.settings as { verticalOffset: number }).verticalOffset).toBe(2);
  });

  it('does not claim readiness for a wrong/ambiguous camera order', () => {
    const { result, worker } = detectSession();
    for (let i = 0; i < 12; i++) {
      act(() => worker.emit({
        kind: 'detect',
        result: { ...detectResult, order: null, agreement: 0.5, matches: 5, medianDisparity: 0 },
      }));
    }
    expect(result.current.status).toBe('failed');
    expect(result.current.order).toBeNull();
    expect(result.current.combineStatus).not.toBe('ready');
  });
});

describe('auto setup profile', () => {
  it('saves the measured profile after a successful auto setup and restores it for the same devices', () => {
    const { result, worker, slots } = detectSession();
    for (let i = 0; i < 3; i++) {
      act(() => worker.emit({ kind: 'detect', result: detectResult }));
    }
    expect(result.current.profileNote).toBe('saved');
    const stored = JSON.parse(localStorage.getItem('stereo-space-profile-v1')!) as SetupProfile;
    expect(stored.aId).toBe('a');
    expect(stored.order).toBe('a-left');

    // Reconnect the SAME devices: the measured geometry comes from the profile.
    const rerender = renderHook((props: ReturnType<typeof options>) => useStereo(props), {
      initialProps: options(makeSlots('a', 'b')),
    });
    void slots;
    expect(rerender.result.current.order).toBe('a-left');
    expect(rerender.result.current.profileNote).toBe('loaded');
    expect(rerender.result.current.status).toBe('automatic');
  });

  it('asks for recalibration when the devices or capture mode changed', () => {
    const { worker } = detectSession();
    for (let i = 0; i < 3; i++) {
      act(() => worker.emit({ kind: 'detect', result: detectResult }));
    }
    expect(localStorage.getItem('stereo-space-profile-v1')).not.toBeNull();

    const rerender = renderHook((props: ReturnType<typeof options>) => useStereo(props), {
      initialProps: options(makeSlots('camera-x', 'camera-y')),
    });
    expect(rerender.result.current.profileNote).toBe('stale');
    expect(rerender.result.current.combineStatus).toBe('setup');
    expect(rerender.result.current.combineStatus).not.toBe('ready');
    expect(localStorage.getItem('stereo-space-profile-v1')).toBeNull();
  });

  it('rejects a setup with insufficient overlap instead of saving a profile', () => {
    const { result, worker } = detectSession();
    for (let i = 0; i < 3; i++) {
      act(() => worker.emit({ kind: 'detect', result: { ...detectResult, overlapRatio: 0.15 } }));
    }
    expect(result.current.order).toBe('a-left');
    expect(result.current.setupAssessment?.ready).toBe(false);
    expect(result.current.setupAssessment?.reason).toBe('low-overlap');
    expect(result.current.combineStatus).toBe('low-overlap');
    expect(localStorage.getItem('stereo-space-profile-v1')).toBeNull();
  });
});

describe('errors and resources', () => {
  it('clears all merging state when a camera is disconnected', () => {
    const { result, worker, rerender } = manualSession();
    const request = worker.messages.find((m) => m.kind === 'reconstruct')!;
    act(() => worker.emit({
      kind: 'reconstruct',
      frameId: request.frameId,
      result: cloud,
      observation: obs(request.frameId as number),
      combined: combinedPayload,
      boxOnCombined,
      combineStatus: 'ready',
    }));
    expect(result.current.combined).not.toBeNull();

    // One camera goes away: nothing may stay on screen as "ready".
    const oneCamera = makeSlots();
    oneCamera.b = null;
    rerender(options(oneCamera));
    expect(result.current.combined).toBeNull();
    expect(result.current.cloud).toBeNull();
    expect(result.current.order).toBeNull();
    expect(result.current.combineStatus).toBe('none');
    expect(result.current.status).toBe('idle');
    expect(worker.terminated).toBe(true);
  });

  it('terminates the worker and stops timers when processing is stopped', () => {
    const { worker, unmount } = manualSession();
    expect(worker.terminated).toBe(false);
    unmount();
    expect(worker.terminated).toBe(true);
  });

  it('ignores messages from a terminated worker after an order swap', () => {
    const { result, worker, request } = (() => {
      const session = manualSession();
      const request = session.worker.messages.find((m) => m.kind === 'reconstruct')!;
      return { ...session, request };
    })();
    act(() => result.current.chooseOrder('b-left'));
    expect(worker.terminated).toBe(true);
    act(() => worker.emit({
      kind: 'reconstruct',
      frameId: request.frameId,
      result: cloud,
      observation: obs(request.frameId as number),
      combined: combinedPayload,
      boxOnCombined,
      combineStatus: 'ready',
    }));
    expect(result.current.combined).toBeNull();
    expect(result.current.combineStatus).toBe('setup');
  });

  it('shows an error status instead of a ready merge when the worker fails', () => {
    const { result, worker } = manualSession();
    act(() => worker.emit({ kind: 'error', message: 'boom' }));
    expect(result.current.combineStatus).toBe('error');
    expect(result.current.combined).toBeNull();
  });
});
