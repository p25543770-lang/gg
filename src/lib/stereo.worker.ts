/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { estimateHamsterDepth, extractRegion, hamsterSearchRegions, mapRegionBox, prepareHamsterInput, selectHamster, type HamsterBox, type HamsterObservation } from './hamster';
import { inferCameraOrder, reconstructStereo, type CameraOrder, type Frame, type StereoSettings } from './stereo';
import { detectCorners, type Corner } from './corners';
import {
  buildDisparityField,
  combineFrames,
  combinePlanar,
  estimateHomography,
  photometricCompensation,
  transferBoxToCombined,
  transferPointToCombined,
  type CombinedFrame,
  type MappedQuad,
} from './stitch';

export type CombineStatus =
  | 'ready'
  | 'planar'
  | 'unreliable-depth'
  | 'low-overlap'
  | 'insufficient'
  | 'error';

/** Serializable subset of CombinedFrame (typed-array buffers are transferred). */
export interface CombinedPayload {
  width: number;
  height: number;
  data: ArrayBuffer;
  coverage: ArrayBuffer;
  mode: 'depth' | 'planar';
  fillRatio: number;
  parallaxRatio: number;
}

export type WorkerRequest =
  | { kind: 'init-hamster'; modelUrl: string }
  | {
    kind: 'detect' | 'reconstruct';
    width: number;
    height: number;
    a: ArrayBuffer;
    b: ArrayBuffer;
    /** Higher-resolution left image captured before downsampling for stereo. */
    detail?: { width: number; height: number; data: ArrayBuffer };
    frameId: number;
    capturedAt: number;
    order?: CameraOrder;
    settings?: StereoSettings;
  };

export type WorkerResponse =
  | { kind: 'detect'; result: ReturnType<typeof inferCameraOrder>; cornersA: Corner[]; cornersB: Corner[] }
  | {
    kind: 'reconstruct';
    result: ReturnType<typeof reconstructStereo>;
    frameId: number;
    observation: HamsterObservation | null;
    combined: CombinedPayload | null;
    boxOnCombined: MappedQuad | null;
    combineStatus: CombineStatus;
    cornersA: Corner[];
    cornersB: Corner[];
    /** Left-frame corners transferred into the combined view by the same warp. */
    cornersCombined: Array<{ x: number; y: number }> | null;
  }
  | { kind: 'hamster-status'; status: 'ready' | 'error'; message?: string }
  | { kind: 'error'; message: string };

let model: ort.InferenceSession | null = null;
let modelStarted = false;
let searchCursor = 0;

async function initHamster(modelUrl: string) {
  if (modelStarted) return;
  modelStarted = true;
  try {
    // One WASM thread works without cross-origin isolation and never needs a CDN.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { wasm: wasmUrl };
    model = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    self.postMessage({ kind: 'hamster-status', status: 'ready' } satisfies WorkerResponse);
  } catch (error) {
    self.postMessage({ kind: 'hamster-status', status: 'error', message: error instanceof Error ? error.message : 'Detector initialization failed' } satisfies WorkerResponse);
  }
}

async function runDetector(session: ort.InferenceSession, pixels: Uint8ClampedArray, width: number, height: number): Promise<HamsterBox | null> {
  const tensor = new ort.Tensor('float32', prepareHamsterInput(pixels, width, height), [1, 300, 300, 3]);
  const output = await session.run({ 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0': tensor });
  const boxes = output['Postprocessor/raw_box_locations:0']?.data;
  const scores = output['Postprocessor/raw_box_scores:0']?.data;
  if (!(boxes instanceof Float32Array) || !(scores instanceof Float32Array) || scores.length !== 1917) throw new Error('Unexpected detector output');
  return selectHamster(boxes, scores);
}

async function analyseHamster(left: Frame, result: ReturnType<typeof reconstructStereo>, request: Extract<WorkerRequest, { kind: 'detect' | 'reconstruct' }>): Promise<HamsterObservation | null> {
  if (!model || !request.order) return null;
  try {
    let box = await runDetector(model, left.data, left.width, left.height);
    // If the animal is small, search overlapping regions of this *same* frame.
    // Time-budget the scan on slow devices and start at the last successful
    // region next time; a result is never combined with a different frame's depth.
    if (!box) {
      const regions = hamsterSearchRegions(left.width, left.height);
      const start = performance.now();
      const first = searchCursor;
      for (let i = 0; i < regions.length; i++) {
        const index = (first + i) % regions.length;
        const region = regions[index];
        const tile = extractRegion(left.data, left.width, left.height, region);
        const local = await runDetector(model, tile, region.width, region.height);
        const mapped = local && mapRegionBox(local, region, left.width, left.height);
        if (mapped) {
          box = mapped;
          searchCursor = index;
          break;
        }
        searchCursor = (index + 1) % regions.length;
        if (performance.now() - start >= 1200) break;
      }
    }
    return {
      frameId: request.frameId,
      capturedAt: request.capturedAt,
      leftSlot: request.order === 'a-left' ? 'a' : 'b',
      width: left.width,
      height: left.height,
      pixels: left.data,
      box,
      distance: box ? estimateHamsterDepth(box, result) : null,
    };
  } catch (error) {
    // Recognition failure must not stop the stereo reconstruction.
    const failed = model;
    model = null;
    void failed?.release();
    self.postMessage({ kind: 'hamster-status', status: 'error', message: error instanceof Error ? error.message : 'Detector inference failed' } satisfies WorkerResponse);
    return null;
  }
}

interface CombineOutcome {
  combined: CombinedPayload | null;
  boxOnCombined: MappedQuad | null;
  cornersCombined: Array<{ x: number; y: number }> | null;
  combineStatus: CombineStatus;
  transfer: Transferable[];
}

function toPayload(frame: CombinedFrame, transfer: Transferable[]): CombinedPayload {
  const data = frame.data.buffer as ArrayBuffer;
  const coverage = frame.coverage.buffer as ArrayBuffer;
  transfer.push(data, coverage);
  return {
    width: frame.width,
    height: frame.height,
    data,
    coverage,
    mode: frame.mode,
    fillRatio: frame.fillRatio,
    parallaxRatio: frame.parallaxRatio,
  };
}

/**
 * Merge the rectified-ish pair into one combined frame. Depth-aware mid-view
 * reprojection when disparity is trustworthy; a labelled planar homography for
 * mostly planar scenes; refusal (raw streams upstream) when nothing is reliable.
 */
function combinePair(
  left: Frame,
  right: Frame,
  settings: StereoSettings,
  box: HamsterBox | null,
  leftCorners: Corner[],
): CombineOutcome {
  const transfer: Transferable[] = [];
  const field = buildDisparityField(left, right, settings.verticalOffset, Math.floor(left.width * 0.3));
  const photometric = photometricCompensation(left, right, field.inlierMatches);

  if (field.matchCount < 12) {
    return { combined: null, boxOnCombined: null, cornersCombined: null, combineStatus: 'insufficient', transfer };
  }

  let frame: CombinedFrame;
  let status: CombineStatus;
  if (field.confidentRatio >= 0.12) {
    frame = combineFrames(left, right, field, {
      maxDisparity: Math.floor(left.width * 0.3),
      verticalOffset: settings.verticalOffset,
    });
    status = field.confidentRatio >= 0.35 ? 'ready' : 'unreliable-depth';
  } else {
    const points = field.inlierMatches.map((m) => ({ from: { x: m.xr, y: m.yr }, to: { x: m.x, y: m.y } }));
    const fit = estimateHomography(points);
    if (!fit) {
      return { combined: null, boxOnCombined: null, cornersCombined: null, combineStatus: 'insufficient', transfer };
    }
    frame = combinePlanar(left, right, fit.h, photometric);
    status = 'planar';
  }
  return {
    combined: toPayload(frame, transfer),
    boxOnCombined: box ? transferBoxToCombined(box, frame.mapping) : null,
    // Corner marks follow the image through the very same geometric warp.
    cornersCombined: leftCorners.length ? leftCorners.map((corner) => transferPointToCombined(corner, frame.mapping)) : [],
    combineStatus: status,
    transfer,
  };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  try {
    const request = event.data;
    if (request.kind === 'init-hamster') {
      void initHamster(request.modelUrl);
      return;
    }
    const frameA: Frame = { width: request.width, height: request.height, data: new Uint8ClampedArray(request.a) };
    const frameB: Frame = { width: request.width, height: request.height, data: new Uint8ClampedArray(request.b) };
    // Both cameras look for corners on their own frames every cycle.
    const cornersA = detectCorners(frameA);
    const cornersB = detectCorners(frameB);
    if (request.kind === 'detect') {
      self.postMessage({ kind: 'detect', result: inferCameraOrder(frameA, frameB), cornersA, cornersB } satisfies WorkerResponse);
    } else {
      if (!request.order || !request.settings) throw new Error('Missing stereo settings');
      const left = request.order === 'a-left' ? frameA : frameB;
      const right = request.order === 'a-left' ? frameB : frameA;
      const leftCorners = request.order === 'a-left' ? cornersA : cornersB;
      const result = reconstructStereo(left, right, request.settings);
      // The detailed detector image and the stereo left image originate from
      // the same captured 4:3 snapshot; only their resolution differs.
      const detectorFrame: Frame = request.detail
        ? { width: request.detail.width, height: request.detail.height, data: new Uint8ClampedArray(request.detail.data) }
        : left;
      const observation = await analyseHamster(detectorFrame, result, request);
      const combined = combinePair(left, right, request.settings, observation?.box ?? null, leftCorners);
      const transfer: Transferable[] = [
        result.positions.buffer,
        result.rgbColors.buffer,
        result.depthColors.buffer,
        result.depthMap.buffer,
        result.pixelIndices.buffer,
        ...combined.transfer,
      ];
      if (observation) transfer.push(observation.pixels.buffer as ArrayBuffer);
      self.postMessage({
        kind: 'reconstruct',
        result,
        frameId: request.frameId,
        observation,
        combined: combined.combined,
        boxOnCombined: combined.boxOnCombined,
        combineStatus: combined.combineStatus,
        cornersA,
        cornersB,
        cornersCombined: combined.cornersCombined,
      } satisfies WorkerResponse, { transfer });
    }
  } catch (error) {
    self.postMessage({ kind: 'error', message: error instanceof Error ? error.message : 'Stereo processing failed' } satisfies WorkerResponse);
  }
};
