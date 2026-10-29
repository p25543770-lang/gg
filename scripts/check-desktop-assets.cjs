const { readFileSync, readdirSync, statSync } = require('node:fs');
const { createHash } = require('node:crypto');
const assert = require('node:assert/strict');
const model = readFileSync('dist/models/hamster-ssd-mobilenet-v2.onnx');
assert.equal(createHash('sha256').update(model).digest('hex'), 'd401301db53697bca98dfefb2425e73c32b1feff86b81574a73ac8ae6670b1f5');
assert(readdirSync('dist/assets').some(name => name.endsWith('.wasm') && statSync(`dist/assets/${name}`).size > 1000000), 'Missing local WASM');
for (const file of ['dist/index.html', 'dist/licenses/Apache-2.0.txt', 'dist/licenses/ONNX-Runtime-MIT.txt']) assert(statSync(file).size > 0);
console.log('Desktop assets: pinned ONNX, WASM, HTML and model/runtime licenses present.');
