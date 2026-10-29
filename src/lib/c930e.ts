/**
 * Logitech C930e stereo pair profile and the classic stereo-vision math from
 * the TrashRobotics "Как Увидеть Глубину? Стереозрение" pipeline:
 *
 *   disparity d = x_left - x_right (the pixel shift of the same object),
 *   depth Z = f * B / d  (similar triangles: f — focal in pixels, known from
 *   the camera's field of view; B — baseline between lens centres, measured).
 *
 * Official Logitech C930e specs used here: 90° diagonal FOV, 82.1° horizontal,
 * 52.2° vertical (16:9), 1080p30 / 720p30, autofocus. Two identical cameras
 * with a fixed lens make a good stereo pair once the streams are synchronised.
 */

export interface CameraOption {
  id: string;
  label: string;
}

export const C930E = {
  label: 'Logitech C930e',
  /** Diagonal field of view, degrees (official datasheet). */
  diagonalFovDeg: 90,
  /** Horizontal field of view at 16:9, degrees (official datasheet). */
  horizontalFovDeg: 82.1,
  /** Vertical field of view at 16:9, degrees (official datasheet). */
  verticalFovDeg: 52.2,
  maxWidth: 1920,
  maxHeight: 1080,
  maxFrameRate: 30,
} as const;

/** Matches "Logitech C930e", "HD Pro Webcam C930e" and similar UVC labels. */
export function isC930e(label: string | undefined | null): boolean {
  return !!label && /c930e/i.test(label);
}

/**
 * Horizontal FOV of a central crop with the given aspect from a full-frame FOV.
 * The app processes a central 4:3 crop, which narrows the 82.1° full view.
 */
export function centralCropFovDeg(fullHorizontalDeg: number, fullVerticalDeg: number, cropAspect: number): number {
  const tanHalfV = Math.tan((fullVerticalDeg * Math.PI) / 360);
  // The crop keeps the full height, so its tan-half-height is tanHalfV and its
  // tan-half-width follows the requested aspect ratio. fullHorizontalDeg is
  // accepted for call-site clarity; the vertical FOV anchors the crop.
  void fullHorizontalDeg;
  return (2 * Math.atan(tanHalfV * cropAspect) * 180) / Math.PI;
}

/** Horizontal FOV of the app's central 4:3 processing crop for a C930e. */
export const C930E_CROP_FOV_DEG = centralCropFovDeg(C930E.horizontalFovDeg, C930E.verticalFovDeg, 4 / 3);

/**
 * Focal length in pixels for an image of the given width: f = (W/2) / tan(hFOV/2).
 * This is the "F" of the stereo-depth formula; it scales with resolution.
 */
export function focalPixels(width: number, horizontalFovDeg: number): number {
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(horizontalFovDeg) || horizontalFovDeg <= 0 || horizontalFovDeg >= 180) {
    throw new Error('Invalid focal parameters');
  }
  return width / (2 * Math.tan((horizontalFovDeg * Math.PI) / 360));
}

/**
 * Depth from disparity exactly as in the video: Z = f * B / d.
 * Disparity must be a positive pixel shift; zero/negative returns null —
 * no distance can be measured without a measurable shift.
 */
export function depthFromDisparity(focalPx: number, baselineM: number, disparityPx: number): number | null {
  if (!Number.isFinite(focalPx) || focalPx <= 0 || !Number.isFinite(baselineM) || baselineM <= 0) return null;
  if (!Number.isFinite(disparityPx) || disparityPx < 0.5) return null;
  return (focalPx * baselineM) / disparityPx;
}

/** Inverse of the depth formula: d = f * B / Z. */
export function disparityFromDepth(focalPx: number, baselineM: number, depthM: number): number | null {
  if (!Number.isFinite(focalPx) || focalPx <= 0 || !Number.isFinite(baselineM) || baselineM <= 0 || !Number.isFinite(depthM) || depthM <= 0) return null;
  return (focalPx * baselineM) / depthM;
}

/**
 * Capture constraints for one C930e. 720p30 at 16:9 is the reliable mode for
 * TWO cameras on USB 2.0 bandwidth (1080p30 x2 often drops frames, which would
 * break the synchronisation the video calls the one hard requirement). The
 * browser falls back automatically if the camera refuses the ideal mode.
 */
export function c930eConstraints(deviceId?: string): MediaStreamConstraints {
  return {
    audio: false,
    video: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      width: { ideal: 1280 },
      height: { ideal: 720 },
      aspectRatio: { ideal: 16 / 9 },
      frameRate: { ideal: 30, max: 30 },
    },
  };
}

/**
 * Auto-picking for the stereo pair: prefer two C930e (the geometry defaults are
 * calculated for them), otherwise the first two different cameras.
 */
export function pickStereoDevices(devices: CameraOption[]): { first: CameraOption; second: CameraOption | null; bothC930e: boolean } | null {
  const unique = devices.filter((device, index) => device.id && devices.findIndex((other) => other.id === device.id) === index);
  if (unique.length < 2) {
    return unique.length === 1 ? { first: unique[0], second: null, bothC930e: false } : null;
  }
  const c930e = unique.filter((device) => isC930e(device.label));
  if (c930e.length >= 2) return { first: c930e[0], second: c930e[1], bothC930e: true };
  return { first: unique[0], second: unique.find((device) => device.id !== unique[0].id) ?? null, bothC930e: false };
}

export interface StereoGeometry {
  /** Focal length in pixels at the processed frame width. */
  focalPx: number;
  /** Baseline in metres (the one measured physical value). */
  baselineM: number;
  /** Working horizontal FOV, degrees. */
  horizontalFovDeg: number;
}

/** Live geometry for the Z = f * B / d readout. */
export function describeGeometry(width: number, horizontalFovDeg: number, baselineCm: number): StereoGeometry {
  return {
    focalPx: focalPixels(width, horizontalFovDeg),
    baselineM: baselineCm / 100,
    horizontalFovDeg,
  };
}
