const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { resolveAsset, isLocal, CSP } = require('../electron/local-files.cjs');
const root = path.resolve('dist');
test('serves only bundled assets at the secure app origin', () => {
  assert.equal(resolveAsset(root, 'stereo://app/'), path.join(root, 'index.html'));
  assert.equal(resolveAsset(root, 'stereo://app/models/hamster-ssd-mobilenet-v2.onnx'), path.join(root, 'models/hamster-ssd-mobilenet-v2.onnx'));
  for (const url of ['https://example.com/a', 'stereo://evil/a', 'stereo://app/%2e%2e%2fsecret', 'stereo://app/%5csecret', 'stereo://app/%00', 'stereo://app/%ZZ']) assert.equal(resolveAsset(root, url), null);
  assert.equal(isLocal('stereo://user@app/'), false);
  assert(CSP.includes("connect-src 'self'"));
});
