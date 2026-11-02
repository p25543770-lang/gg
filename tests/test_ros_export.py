"""Checks for the UDP telemetry export consumed by the ROS 2 bridge."""
import json
import socket
import unittest

from pose import KEYPOINT_NAMES, PersonPose, PoseInfo
from ros_export import PAYLOAD_VERSION, TelemetrySender, build_payload, valid_host

import ros2_bridge


class FakePerson:
    """Stand-in for PersonPose without requiring the ONNX models."""

    def __init__(self, track_id, bbox, score, keypoints):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.keypoints = keypoints


class FakeRobotInfo:
    def __init__(self):
        import robot_vision

        self.info = robot_vision.RobotInfo(road_fraction=0.42, obstacle_count=1,
                                           center_blocked=True,
                                           tracks=[robot_vision.ObstacleTrack(3, (10, 20, 60, 80), (35, 50))])


class FakePoseInfo(PoseInfo):
    def __init__(self, people, backend='ONNX Runtime'):
        self.people = people
        self.inference_ms = 12.3
        self.status = 'PEOPLE: 1'
        self.detected = False
        self.backend = backend


class BuildPayloadTests(unittest.TestCase):
    def test_payload_shape_and_values(self):
        keypoints = [[float(i), float(i) * 2, 0.9] for i in range(len(KEYPOINT_NAMES))]
        person = FakePerson(2, (5.0, 6.0, 105.0, 206.0), 0.69, keypoints)
        payload = build_payload(1, 'skeleton', (640, 480), 7,
                                robot_info=FakeRobotInfo().info,
                                pose_info=FakePoseInfo([person]))
        self.assertEqual(payload['v'], PAYLOAD_VERSION)
        self.assertEqual(payload['camera'], 1)
        self.assertEqual(payload['seq'], 7)
        self.assertEqual(payload['mode'], 'skeleton')
        self.assertEqual(payload['frame'], {'w': 640, 'h': 480})
        # json round trip must work: this exact dict goes into one UDP packet.
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIsInstance(encoded, str)
        decoded = json.loads(encoded)
        self.assertEqual(len(decoded['people'][0]['keypoints']), 17)
        self.assertEqual(decoded['people'][0]['keypoints'][0], [0.0, 0.0, 0.9])
        self.assertEqual(decoded['people'][0]['side'], 'L')
        self.assertEqual(decoded['obstacles'][0]['id'], 3)
        self.assertTrue(decoded['road']['center_blocked'])
        self.assertEqual(decoded['engine'], 'ONNX Runtime')

    def test_empty_mode_still_sends_a_heartbeat(self):
        payload = build_payload(0, 'edges', (320, 240), 1)
        self.assertEqual(payload['obstacles'], [])
        self.assertEqual(payload['people'], [])
        self.assertNotIn('road', payload)
        self.assertNotIn('ms', payload)


class TelemetrySenderTests(unittest.TestCase):
    def test_delivery_over_real_udp_loopback(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(2.0)
        port = receiver.getsockname()[1]
        sender = TelemetrySender('127.0.0.1', port)
        payload = build_payload(0, 'robot', (64, 64), 1, robot_info=FakeRobotInfo().info)
        sender.send(payload)
        data, _ = receiver.recvfrom(65536)
        decoded = json.loads(data.decode('utf-8'))
        self.assertEqual(decoded['obstacles'][0]['id'], 3)
        self.assertEqual(sender.sent, 1)
        self.assertEqual(sender.errors, 0)
        sender.close()
        receiver.close()

    def test_network_errors_are_counted_not_raised(self):
        # Port 9 (discard) usually has no listener here; any closed port works
        # because UDP errors surface on the next call at most.
        sender = TelemetrySender('255.255.255.255', 9)
        sender.send(build_payload(0, 'edges', (8, 8), 1))
        self.assertEqual(sender.sent + sender.errors, 1)
        sender.close()

    def test_host_validation(self):
        self.assertTrue(valid_host('127.0.0.1'))
        self.assertTrue(valid_host('192.168.1.10'))
        self.assertFalse(valid_host('robot'))
        self.assertFalse(valid_host(''))


class BridgeContractTests(unittest.TestCase):
    """The bridge file must work without ROS 2 installed (pure parsing)."""

    def test_parse_packet(self):
        good = json.dumps({'v': 1, 'camera': 0, 'mode': 'skeleton', 'people': []}).encode('utf-8')
        self.assertEqual(ros2_bridge.parse_packet(good)['camera'], 0)
        self.assertIsNone(ros2_bridge.parse_packet(b'not json'))
        self.assertIsNone(ros2_bridge.parse_packet(json.dumps({'v': 2}).encode('utf-8')))
        self.assertIsNone(ros2_bridge.parse_packet(json.dumps({'v': 1}).encode('utf-8')))

    def test_people_and_obstacles_extraction(self):
        packet = {'v': 1, 'camera': 0, 'mode': 'skeleton',
                  'people': [{'id': 1, 'box': [0, 0, 10, 20], 'score': 0.5,
                              'keypoints': [1.0] * (17 * 3)},
                             {'id': 2, 'box': [0, 0, 1, 1], 'score': 0.2,
                              'keypoints': [1.0]}]}
        people = ros2_bridge.people_from_packet(packet)
        self.assertEqual(len(people), 1)
        self.assertEqual(len(people[0]['keypoints']), 17 * 3)
        obstacles = ros2_bridge.obstacles_from_packet(
            {'obstacles': [{'id': 5, 'box': [1, 2, 3, 4]}, {'id': 6}]})
        self.assertEqual(obstacles[0]['id'], 5)
        self.assertEqual(obstacles[1]['id'], 6)

    def test_skeleton_layout_constant(self):
        self.assertEqual(ros2_bridge.SKELETON_LAYOUT, 7 + 17 * 3)


if __name__ == '__main__':
    unittest.main()
