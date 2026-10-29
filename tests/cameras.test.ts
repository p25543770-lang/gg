// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useCameras } from '../src/hooks/useCameras';

class FakeTrack {
  readyState: 'live' | 'ended' = 'live';
  label: string;
  private endedListener?: () => void;
  constructor(public id: string) { this.label = `Camera ${id}`; }
  getSettings() { return { deviceId: this.id }; }
  addEventListener(type: string, listener: () => void) { if (type === 'ended') this.endedListener = listener; }
  stop = vi.fn(() => { this.readyState = 'ended'; });
  end() { this.readyState = 'ended'; this.endedListener?.(); }
}

class FakeStream {
  track: FakeTrack;
  constructor(id: string) { this.track = new FakeTrack(id); }
  getVideoTracks() { return [this.track]; }
  getTracks() { return [this.track]; }
}

function device(id: string) {
  return { kind: 'videoinput', deviceId: id, label: `Camera ${id}`, groupId: id, toJSON: () => ({}) };
}

let available: ReturnType<typeof device>[];
let streams: FakeStream[];
let media: {
  getUserMedia: ReturnType<typeof vi.fn>;
  enumerateDevices: ReturnType<typeof vi.fn>;
  addEventListener: ReturnType<typeof vi.fn>;
  removeEventListener: ReturnType<typeof vi.fn>;
};

beforeEach(() => {
  available = [device('a'), device('b')];
  streams = [];
  media = {
    getUserMedia: vi.fn(async (constraints: MediaStreamConstraints) => {
      const video = constraints.video as MediaTrackConstraints;
      const deviceId = video.deviceId as { exact?: string } | undefined;
      const stream = new FakeStream(deviceId?.exact ?? 'a');
      streams.push(stream);
      return stream as unknown as MediaStream;
    }),
    enumerateDevices: vi.fn(async () => available),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: media });
});

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('camera lifecycle', () => {
  it('opens two different devices and stops both on disconnect', async () => {
    const { result } = renderHook(() => useCameras());
    await act(async () => { await result.current.connect(); });
    expect(result.current.slots.a?.id).toBe('a');
    expect(result.current.slots.b?.id).toBe('b');
    expect(media.getUserMedia).toHaveBeenCalledTimes(2);
    expect(result.current.error).toBeNull();
    act(() => result.current.disconnect());
    expect(streams.every((stream) => stream.track.stop.mock.calls.length === 1)).toBe(true);
    expect(result.current.slots).toEqual({ a: null, b: null });
  });

  it('keeps one camera usable and allows choosing a newly attached second one', async () => {
    available = [device('a')];
    const { result } = renderHook(() => useCameras());
    await act(async () => { await result.current.connect(); });
    expect(result.current.slots.a?.id).toBe('a');
    expect(result.current.slots.b).toBeNull();
    expect(result.current.error).toContain('только одна камера');
    available = [device('a'), device('b')];
    await act(async () => { await result.current.refresh(); });
    expect(result.current.devices).toHaveLength(2);
    await act(async () => { await result.current.changeCamera('b', 'b'); });
    expect(result.current.slots.b?.id).toBe('b');
    expect(result.current.error).toBeNull();
  });

  it('keeps the old stream if opening a replacement fails', async () => {
    available.push(device('c'));
    const { result } = renderHook(() => useCameras());
    await act(async () => { await result.current.connect(); });
    const oldB = result.current.slots.b?.stream;
    media.getUserMedia.mockRejectedValueOnce(new DOMException('busy', 'NotReadableError'));
    await act(async () => { await result.current.changeCamera('b', 'c'); });
    expect(result.current.slots.b?.stream).toBe(oldB);
    expect(streams[1].track.readyState).toBe('live');
    expect(result.current.error).toContain('занята');
  });

  it('stops a replaced stream and prevents choosing the same device twice', async () => {
    available.push(device('c'));
    const { result } = renderHook(() => useCameras());
    await act(async () => { await result.current.connect(); });
    await act(async () => { await result.current.changeCamera('b', 'c'); });
    expect(result.current.slots.b?.id).toBe('c');
    expect(streams[1].track.stop).toHaveBeenCalledOnce();
    await act(async () => { await result.current.changeCamera('b', 'a'); });
    expect(result.current.slots.b?.id).toBe('c');
    expect(result.current.error).toContain('две разные камеры');
  });

  it('clears a stream that the browser ends while opening the second camera', async () => {
    const first = new FakeStream('a');
    media.getUserMedia.mockResolvedValueOnce(first as unknown as MediaStream).mockImplementationOnce(async () => {
      first.track.readyState = 'ended';
      throw new DOMException('busy', 'NotReadableError');
    });
    const { result } = renderHook(() => useCameras());
    await act(async () => { await result.current.connect(); });
    expect(result.current.slots.a).toBeNull();
    expect(result.current.slots.b).toBeNull();
    expect(first.track.stop).toHaveBeenCalled();
    expect(result.current.error).toContain('занята');
  });

  it('handles permission denial and device removal', async () => {
    media.getUserMedia.mockRejectedValueOnce(new DOMException('denied', 'NotAllowedError'));
    const { result } = renderHook(() => useCameras());
    await act(async () => { await result.current.connect(); });
    expect(result.current.error).toContain('Разрешите');
    expect(result.current.busy).toBe(false);
    await act(async () => { await result.current.connect(); });
    act(() => streams[0].track.end());
    expect(result.current.slots.a).toBeNull();
    expect(result.current.error).toContain('отключилась');
  });

  it('stops a stream whose permission request resolves after unmount', async () => {
    let resolve!: (stream: MediaStream) => void;
    media.getUserMedia.mockReturnValueOnce(new Promise<MediaStream>((done) => { resolve = done; }));
    const { result, unmount } = renderHook(() => useCameras());
    let pending!: Promise<void>;
    act(() => { pending = result.current.connect(); });
    unmount();
    const late = new FakeStream('a');
    await act(async () => { resolve(late as unknown as MediaStream); await pending; });
    expect(late.track.stop).toHaveBeenCalledOnce();
  });
});
