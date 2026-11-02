"""UDP telemetry export for a ROS 2 robot.

The Windows app stays a plain standalone executable: it only sends small JSON
datagrams with what it currently sees. A tiny bridge script (ros2_bridge.py,
shipped in the repository) runs on the robot's onboard computer, receives the
datagrams and republishes them as standard ROS 2 topics. No ROS 2 install, no
extra wheels and no network libraries are needed on the Windows side — just
the standard library.
"""
import json
import socket
import time

from pose import KEYPOINT_NAMES

PAYLOAD_VERSION = 1
# Number of numbers per person in the skeleton export: bbox (4) + score (1).
SKELETON_HEADER = 5


class TelemetrySender:
    """Fire-and-forget UDP sender. Network errors are counted, never raised."""

    def __init__(self, host, port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._address = (host, int(port))
        self.sent = 0
        self.errors = 0

    def send(self, payload):
        try:
            data = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            self._socket.sendto(data, self._address)
            self.sent += 1
        except OSError:
            self.errors += 1

    def close(self):
        self._socket.close()


def _side(x1, x2, width):
    center = (x1 + x2) / 2.0
    return 'L' if center < width * 0.38 else ('R' if center > width * 0.62 else 'C')


def _box(bbox):
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def build_payload(camera, mode, frame_size, seq, robot_info=None, pose_info=None, stamp=None):
    """One JSON-serialisable snapshot of what one camera currently shows.

    Coordinates are pixels in the processed frame; ``frame`` carries the size
    so the robot can normalise them. ``keypoints`` follows the COCO-17 order
    of KEYPOINT_NAMES.
    """
    width, height = int(frame_size[0]), int(frame_size[1])
    payload = {
        'v': PAYLOAD_VERSION,
        'camera': int(camera),
        'seq': int(seq),
        'stamp': round(stamp if stamp is not None else time.time(), 3),
        'mode': str(mode),
        'frame': {'w': width, 'h': height},
        'obstacles': [],
        'people': [],
    }
    if robot_info is not None:
        payload['road'] = {'fraction': round(float(robot_info.road_fraction), 3),
                           'center_blocked': bool(robot_info.center_blocked)}
        for track in robot_info.tracks:
            x1, y1, x2, y2 = (float(v) for v in track.bbox)
            payload['obstacles'].append({
                'id': int(track.track_id),
                'box': _box(track.bbox),
                'side': _side(x1, x2, width),
                'cx': round((x1 + x2) / 2.0, 1),
                'cy': round((y1 + y2) / 2.0, 1),
            })
    if pose_info is not None:
        payload['ms'] = round(float(pose_info.inference_ms), 1)
        payload['engine'] = pose_info.backend
        for person in pose_info.people:
            x1, y1, x2, y2 = (float(v) for v in person.bbox)
            keypoints = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                         for x, y, c in person.keypoints[:len(KEYPOINT_NAMES)]]
            payload['people'].append({
                'id': int(person.track_id),
                'box': _box(person.bbox),
                'side': _side(x1, x2, width),
                'score': round(float(person.score), 3),
                'keypoints': keypoints,
            })
    return payload


def valid_host(host):
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return False
