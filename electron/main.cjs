const { app, BrowserWindow, dialog, protocol, session } = require('electron');
const path = require('node:path');
const { readFile } = require('node:fs/promises');
const { ORIGIN, CSP, isLocal, resolveAsset } = require('./local-files.cjs');

protocol.registerSchemesAsPrivileged([
  { scheme: 'stereo', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true } },
]);

if (!app.requestSingleInstanceLock()) app.quit();
else {
  let window;
  app.on('second-instance', () => {
    if (window) {
      if (window.isMinimized()) window.restore();
      window.focus();
    }
  });

  app.whenReady().then(async () => {
    const root = path.join(app.getAppPath(), 'dist');
    protocol.handle('stereo', async (request) => {
      const file = resolveAsset(root, request.url);
      if (!file || !['GET', 'HEAD'].includes(request.method)) return new Response(null, { status: 403 });
      try {
        const data = await readFile(file);
        const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.json': 'application/json', '.woff': 'font/woff', '.woff2': 'font/woff2', '.svg': 'image/svg+xml', '.png': 'image/png', '.txt': 'text/plain; charset=utf-8' };
        const headers = new Headers({ 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
        headers.set('Content-Security-Policy', CSP);
        headers.set('X-Content-Type-Options', 'nosniff');
        return new Response(request.method === 'HEAD' ? null : data, { status: 200, headers });
      } catch { return new Response(null, { status: 404 }); }
    });

    const ses = session.defaultSession;
    // No external API/CDN, including requests made by workers.
    ses.webRequest.onBeforeRequest((details, callback) => {
      callback({ cancel: !isLocal(details.url) && !details.url.startsWith('blob:') });
    });
    ses.setPermissionCheckHandler((contents, permission, origin, details) =>
      contents === window?.webContents && isLocal(origin) && permission === 'media' && details.mediaType === 'video');
    ses.setPermissionRequestHandler((contents, permission, callback, details) => {
      callback(contents === window?.webContents && isLocal(details.requestingUrl) &&
        details.isMainFrame && permission === 'media' && details.mediaTypes?.length > 0 &&
        details.mediaTypes.every((type) => type === 'video'));
    });

    window = new BrowserWindow({
      title: 'Stereo Space', width: 1440, height: 960, minWidth: 900, minHeight: 650,
      autoHideMenuBar: true,
      webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true, webSecurity: true },
    });
    window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    window.webContents.on('will-navigate', (event, url) => { if (!isLocal(url)) event.preventDefault(); });
    await window.loadURL(`${ORIGIN}/`);
  }).catch((error) => {
    dialog.showErrorBox('Stereo Space — ошибка запуска', String(error));
    app.quit();
  });
  app.on('window-all-closed', () => app.quit());
}
