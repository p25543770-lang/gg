// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useStereo } from '../src/hooks/useStereo';
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
const slots: CameraSlots = {
  a: { id: 'a', label: 'A', stream: {} as MediaStream },
  b: { id: 'b', label: 'B', stream: {} as MediaStream },
};
const videoA = { current: { videoWidth: 640, videoHeight: 480, readyState: 4 } as HTMLVideoElement };
const videoB = { current: { videoWidth: 640, videoHeight: 480, readyState: 4 } as HTMLVideoElement };

beforeEach(() => {
  vi.useFakeTimers();
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

function manualSession() {
  const hook = renderHook(() => useStereo({ slots, videoA, videoB, active: true, baselineCm: 6.5, horizontalFov: 60, maxDepthM: 8 }));
  act(() => hook.result.current.chooseOrder('a-left'));
  const worker = FakeWorker.instances.at(-1)!;
  expect(worker.messages[0].kind).toBe('init-hamster');
  act(() => { worker.emit({ kind: 'hamster-status', status: 'ready' }); vi.advanceTimersByTime(100); });
  const request = worker.messages.find((message) => message.kind === 'reconstruct')!;
  expect(request).toBeDefined();
  const detail = request.detail as { width: number; height: number; data: ArrayBuffer };
  expect(detail.width).toBe(512);
  expect(detail.height).toBe(384);
  expect(detail.data.byteLength).toBe(512 * 384 * 4);
  expect((request.a as ArrayBuffer).byteLength).toBe(256 * 192 * 4);
  return { ...hook, worker, request };
}

describe('detector / stereo frame lifecycle', () => {
  it('accepts only an observation with the same frame ID as its stereo result', () => {
    const { result, worker, request } = manualSession();
    const id = request.frameId as number;
    act(() => worker.emit({ kind: 'reconstruct', frameId: id, result: cloud, observation: obs(id + 1) }));
    expect(result.current.cloud).toBe(cloud);
    expect(result.current.observation).toBeNull();
    act(() => { vi.advanceTimersByTime(450); });
    const next = worker.messages.filter((message) => message.kind === 'reconstruct').at(-1)!;
    act(() => worker.emit({ kind: 'reconstruct', frameId: next.frameId, result: cloud, observation: obs(next.frameId as number) }));
    expect(result.current.observation?.frameId).toBe(next.frameId);
  });

  it('clears a stale observation and refuses an already-old response', () => {
    const { result, worker, request } = manualSession();
    const id = request.frameId as number;
    act(() => worker.emit({ kind: 'reconstruct', frameId: id, result: cloud, observation: obs(id) }));
    expect(result.current.observation).not.toBeNull();
    act(() => vi.advanceTimersByTime(5001));
    expect(result.current.observation).toBeNull();
    act(() => worker.emit({ kind: 'reconstruct', frameId: id, result: cloud, observation: obs(id, Date.now() - 6000) }));
    expect(result.current.observation).toBeNull();
  });

  it('clears the previous result on swap and ignores messages from the terminated worker', () => {
    const { result, worker, request } = manualSession();
    const id = request.frameId as number;
    act(() => worker.emit({ kind: 'reconstruct', frameId: id, result: cloud, observation: obs(id) }));
    expect(result.current.observation).not.toBeNull();
    act(() => result.current.chooseOrder('b-left'));
    expect(worker.terminated).toBe(true);
    expect(result.current.observation).toBeNull();
    act(() => worker.emit({ kind: 'reconstruct', frameId: id, result: cloud, observation: obs(id) }));
    expect(result.current.observation).toBeNull();
  });

  it('keeps stereo data available if the detector fails and can retry', () => {
    const { result, worker, request } = manualSession();
    act(() => {
      worker.emit({ kind: 'hamster-status', status: 'error' });
      worker.emit({ kind: 'reconstruct', frameId: request.frameId, result: cloud, observation: null });
    });
    expect(result.current.hamsterStatus).toBe('error');
    expect(result.current.cloud).toBe(cloud);
    act(() => result.current.retryHamster());
    expect(worker.terminated).toBe(true);
    expect(result.current.hamsterStatus).toBe('loading');
    expect(FakeWorker.instances.at(-1)?.messages[0].kind).toBe('init-hamster');
  });
});
