#!/usr/bin/env python3
"""Мост Purple Corners → ROS 2. Один файл, запускается на компьютере робота.

Программа PurpleCorners (Windows) отправляет маленькие UDP-пакеты с тем, что
видят камеры. Этот скрипт принимает их и публикует в стандартные топики ROS 2.
Никаких дополнительных пакетов, кроме самого ROS 2, не нужно.

Запуск на роботе (Ubuntu + ROS 2, например Humble):

    python3 ros2_bridge.py --port 7555

Проверка в другом терминале:

    ros2 topic list
    ros2 topic echo /purple_corners/obstacles
    ros2 topic hz /purple_corners/people

Топики (если в ROS 2 есть vision_msgs из common_interfaces — он обычно есть):

    /purple_corners/obstacles   vision_msgs/Detection2DArray  красные рамки препятствий
    /purple_corners/people      vision_msgs/Detection2DArray  рамки людей
    /purple_corners/skeleton    std_msgs/Float32MultiArray    скелеты, см. SKELETON_LAYOUT ниже
    /purple_corners/status      std_msgs/String               режим, камера, время обработки
    /purple_corners/json        std_msgs/String               полный пакет как есть

Запасной вариант: если vision_msgs не найден, всё публикуется как JSON в
/purple_corners/json, а в начале печатается предупреждение.

Схема Float32MultiArray для скелетов (SKELETON_LAYOUT чисел на человека):
    [камера, id, x1, y1, x2, y2, уверенность, затем 17 раз (x, y, видимость)]
Точки идут в порядке COCO-17: нос, глаза, уши, плечи, локти, кисти, бёдра,
колени, ступни. Координаты — пиксели кадра (размер кадра есть в status).

В приложении: блок «Экспорт для ROS 2» внизу окна — включите галочку и укажите
IP робота и порт (по умолчанию 7555). Всё в этой связке — оценка по картинке,
а не система безопасности: не вожди робота этими данными без независимых
датчиков.
"""
import argparse
import json
import queue
import socket
import sys
import threading
import time

TOPIC_PREFIX = '/purple_corners'
PAYLOAD_VERSION = 1
# 7 чисел шапки (камера, id, x1, y1, x2, y2, score) + 17 точек по 3 числа.
SKELETON_HEADER = 7
KEYPOINTS = 17
SKELETON_LAYOUT = SKELETON_HEADER + KEYPOINTS * 3


def parse_packet(data):
    """Decode one UDP datagram; return a dict or None for anything foreign."""
    try:
        packet = json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(packet, dict) or packet.get('v') != PAYLOAD_VERSION:
        return None
    if 'camera' not in packet or 'mode' not in packet:
        return None
    return packet


def people_from_packet(packet):
    """Normalized people list: ints/floats only, missing fields tolerated."""
    result = []
    for person in packet.get('people') or []:
        try:
            keypoints = [float(v) for v in person.get('keypoints', [])]
        except (TypeError, ValueError):
            continue
        if len(keypoints) < KEYPOINTS * 3:
            continue
        box = [float(v) for v in person.get('box', [0, 0, 0, 0])]
        result.append({'id': int(person.get('id', 0)),
                       'box': box,
                       'score': float(person.get('score', 0.0)),
                       'keypoints': keypoints[:KEYPOINTS * 3]})
    return result


def obstacles_from_packet(packet):
    result = []
    for obstacle in packet.get('obstacles') or []:
        try:
            box = [float(v) for v in obstacle.get('box', [0, 0, 0, 0])]
        except (TypeError, ValueError):
            continue
        result.append({'id': int(obstacle.get('id', 0)), 'box': box})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Мост Purple Corners → ROS 2 (UDP JSON → топики).')
    parser.add_argument('--host', default='0.0.0.0', help='адрес для приёма UDP (по умолчанию все интерфейсы)')
    parser.add_argument('--port', type=int, default=7555, help='UDP-порт из блока «Экспорт для ROS 2» (по умолчанию 7555)')
    args = parser.parse_args(argv)

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from std_msgs.msg import Float32MultiArray, String
    except ImportError:
        print('Не найден ROS 2 (модуль rclpy). Запустите этот скрипт на компьютере с ROS 2,')
        print('например: source /opt/ros/humble/setup.bash && python3 ros2_bridge.py')
        return 1

    try:
        from vision_msgs.msg import (BoundingBox2D, Detection2D, Detection2DArray,
                                     ObjectHypothesisWithPose)
        from geometry_msgs.msg import Pose2D
    except ImportError:
        BoundingBox2D = Detection2D = Detection2DArray = ObjectHypothesisWithPose = Pose2D = None
        print('Внимание: vision_msgs не найден — всё публикуется как JSON в '
              f'{TOPIC_PREFIX}/json.')

    rclpy.init()
    node = Node('purple_corners_bridge')
    vision_ready = Detection2DArray is not None

    def detection_array(camera, entries, kind):
        array = Detection2DArray()
        for entry in entries:
            detection = Detection2D()
            detection.id = f'c{camera}#{entry["id"]}'
            x1, y1, x2, y2 = entry['box']
            detection.bbox = BoundingBox2D()
            detection.bbox.center = Pose2D()
            detection.bbox.center.x = (x1 + x2) / 2.0
            detection.bbox.center.y = (y1 + y2) / 2.0
            detection.bbox.size_x = max(0.0, x2 - x1)
            detection.bbox.size_y = max(0.0, y2 - y1)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.id = kind
            hypothesis.hypothesis.score = float(entry.get('score', 1.0))
            detection.results.append(hypothesis)
            array.detections.append(detection)
        return array

    publishers = {
        'obstacles': node.create_publisher(Detection2DArray, TOPIC_PREFIX + '/obstacles',
                                           qos_profile_sensor_data) if vision_ready else None,
        'people': node.create_publisher(Detection2DArray, TOPIC_PREFIX + '/people',
                                        qos_profile_sensor_data) if vision_ready else None,
        'skeleton': node.create_publisher(Float32MultiArray, TOPIC_PREFIX + '/skeleton',
                                          qos_profile_sensor_data),
        'status': node.create_publisher(String, TOPIC_PREFIX + '/status', qos_profile_sensor_data),
        'json': node.create_publisher(String, TOPIC_PREFIX + '/json', qos_profile_sensor_data),
    }

    packets = queue.Queue(maxsize=64)
    stop = threading.Event()

    def listen():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 18)
        try:
            sock.bind((args.host, args.port))
        except OSError as exc:
            packets.put({'__error__': f'Не удалось занять порт {args.port}: {exc}'})
            return
        print(f'Мост слушает UDP {args.host}:{args.port}. Ожидание пакетов из PurpleCorners…')
        while not stop.is_set():
            try:
                data, _address = sock.recvfrom(1 << 16)
            except OSError:
                return
            packet = parse_packet(data)
            if packet is None:
                continue
            try:
                packets.put_nowait(packet)
            except queue.Full:
                pass  # робот не успевает — старые кадры и не нужны

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()

    def publish(packet):
        camera = int(packet['camera'])
        text = String()
        text.data = json.dumps(packet, ensure_ascii=False)
        publishers['json'].publish(text)
        status = String()
        status.data = json.dumps({'camera': camera, 'mode': packet['mode'],
                                  'ms': packet.get('ms'), 'engine': packet.get('engine')},
                                 ensure_ascii=False)
        publishers['status'].publish(status)
        obstacles = obstacles_from_packet(packet)
        people = people_from_packet(packet)
        if vision_ready:
            if obstacles:
                publishers['obstacles'].publish(detection_array(camera, obstacles, 'obstacle'))
            if people:
                publishers['people'].publish(detection_array(camera, people, 'person'))
        for person in people:
            array = Float32MultiArray()
            x1, y1, x2, y2 = person['box']
            array.data = ([float(camera), float(person['id']), x1, y1, x2, y2, person['score']]
                          + person['keypoints'])
            publishers['skeleton'].publish(array)

    printed_error = False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                packet = packets.get_nowait()
            except queue.Empty:
                continue
            if '__error__' in packet:
                if not printed_error:
                    print(packet['__error__'])
                    printed_error = True
                continue
            publish(packet)
    except KeyboardInterrupt:
        print('\nОстановлено.')
    finally:
        stop.set()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
