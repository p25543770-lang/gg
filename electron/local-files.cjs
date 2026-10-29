const path = require('node:path');

const ORIGIN = 'stereo://app';
const CSP = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-src 'none'; base-uri 'none'";

function isLocal(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'stereo:' && parsed.host === 'app' && !parsed.username && !parsed.password;
  } catch { return false; }
}

function resolveAsset(root, url) {
  if (!isLocal(url)) return null;
  try {
    const pathname = decodeURIComponent(new URL(url).pathname);
    if (pathname.includes('\\') || pathname.includes('\0')) return null;
    const file = path.resolve(root, `.${pathname === '/' ? '/index.html' : pathname}`);
    const relative = path.relative(root, file);
    return relative && !relative.startsWith('..') && !path.isAbsolute(relative) ? file : null;
  } catch { return null; }
}

module.exports = { ORIGIN, CSP, isLocal, resolveAsset };
