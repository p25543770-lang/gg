import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-web/wasm';
import { selectHamster } from '../src/lib/hamster';

describe('bundled local detector', () => {
  it('loads the pinned model with local WASM and does not find a hamster in a blank frame', async () => {
    const bytes = await readFile(resolve('public/models/hamster-ssd-mobilenet-v2.onnx'));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe('d401301db53697bca98dfefb2425e73c32b1feff86b81574a73ac8ae6670b1f5');
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ['wasm'] });
    try {
      const output = await session.run({
        'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0': new ort.Tensor('float32', new Float32Array(300 * 300 * 3), [1, 300, 300, 3]),
      });
      const boxes = output['Postprocessor/raw_box_locations:0'].data as Float32Array;
      const scores = output['Postprocessor/raw_box_scores:0'].data as Float32Array;
      expect(scores.length).toBe(1917);
      expect(selectHamster(boxes, scores)).toBeNull();
    } finally {
      await session.release();
    }
  }, 30000);
});
