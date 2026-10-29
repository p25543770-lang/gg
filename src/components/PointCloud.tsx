import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { PointCloudData } from '../lib/stereo';

export interface PointCloudHandle {
  reset: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  toggleGrid: () => boolean;
}

interface Props {
  cloud: PointCloudData | null;
  colorMode: 'depth' | 'rgb';
  sceneKey: string;
}

interface SceneObjects {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  points: THREE.Points;
  grid: THREE.GridHelper;
  texture: THREE.CanvasTexture;
}

function pointTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const context = canvas.getContext('2d')!;
  const gradient = context.createRadialGradient(16, 16, 2, 16, 16, 15);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.55, 'rgba(255,255,255,0.95)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 32, 32);
  return new THREE.CanvasTexture(canvas);
}

function fitLiveCloud(scene: SceneObjects, cloud: PointCloudData) {
  if (!cloud.count) return;
  // Percentiles keep occasional mismatches from pushing the useful scene out
  // of view. Framing happens once per source, not on every incoming frame.
  const axes: number[][] = [[], [], []];
  for (let i = 0; i < cloud.positions.length; i += 3) {
    axes[0].push(cloud.positions[i]);
    axes[1].push(cloud.positions[i + 1]);
    axes[2].push(cloud.positions[i + 2]);
  }
  axes.forEach((axis) => axis.sort((a, b) => a - b));
  const at = (axis: number[], percentile: number) => axis[Math.floor((axis.length - 1) * percentile)];
  const min = new THREE.Vector3(...axes.map((axis) => at(axis, 0.05)) as [number, number, number]);
  const max = new THREE.Vector3(...axes.map((axis) => at(axis, 0.95)) as [number, number, number]);
  const size = max.clone().sub(min);
  const center = min.clone().add(max).multiplyScalar(0.5);
  const extent = Math.max(size.x, size.y, size.z, 0.25);
  const visibleHeight = Math.max(size.y, size.x / Math.max(scene.camera.aspect, 0.5), size.z * 0.7, 0.25);
  const distance = Math.max(0.65, visibleHeight / (2 * Math.tan(THREE.MathUtils.degToRad(scene.camera.fov / 2))) * 1.8);
  const direction = new THREE.Vector3(0.7, 0.5, 1).normalize();
  scene.controls.target.copy(center);
  scene.camera.position.copy(center).addScaledVector(direction, distance);
  scene.controls.minDistance = Math.max(0.1, extent * 0.08);
  scene.controls.maxDistance = Math.max(30, extent * 15);
  scene.grid.position.set(center.x, min.y - Math.max(0.1, extent * 0.15), center.z);
  scene.grid.scale.setScalar(Math.max(0.16, extent / 5));
  scene.controls.update();
  scene.controls.saveState();
}

const PointCloud = forwardRef<PointCloudHandle, Props>(function PointCloud({ cloud, colorMode, sceneKey }, ref) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<SceneObjects | null>(null);
  const lastSceneRef = useRef<string | null>(null);
  const currentCloudRef = useRef<PointCloudData | null>(cloud);
  const currentSceneKeyRef = useRef(sceneKey);
  currentCloudRef.current = cloud;
  currentSceneKeyRef.current = sceneKey;
  const [webGlError, setWebGlError] = useState(false);

  useImperativeHandle(ref, () => ({
    reset: () => {
      const scene = sceneRef.current;
      if (!scene) return;
      if (currentSceneKeyRef.current !== 'demo' && currentCloudRef.current?.count) fitLiveCloud(scene, currentCloudRef.current);
      else scene.controls.reset();
    },
    zoomIn: () => {
      const scene = sceneRef.current;
      if (!scene) return;
      scene.camera.position.sub(scene.controls.target).multiplyScalar(0.78).add(scene.controls.target);
      scene.controls.update();
    },
    zoomOut: () => {
      const scene = sceneRef.current;
      if (!scene) return;
      scene.camera.position.sub(scene.controls.target).multiplyScalar(1.28).add(scene.controls.target);
      scene.controls.update();
    },
    toggleGrid: () => {
      const grid = sceneRef.current?.grid;
      if (!grid) return false;
      grid.visible = !grid.visible;
      return grid.visible;
    },
  }), []);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    } catch {
      setWebGlError(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(43, 1, 0.05, 100);
    camera.position.set(4.9, 3.8, 5.1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, -3.1);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.minDistance = 0.5;
    controls.maxDistance = 30;
    controls.maxPolarAngle = Math.PI * 0.88;
    controls.update();
    controls.saveState();

    const grid = new THREE.GridHelper(14, 28, 0x385253, 0x263d3e);
    grid.position.set(0, -1.09, -3);
    const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
    gridMaterials.forEach((material) => { material.transparent = true; material.opacity = 0.48; });
    scene.add(grid);

    const texture = pointTexture();
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.PointsMaterial({
      size: 0.065,
      vertexColors: true,
      map: texture,
      transparent: true,
      opacity: 0.94,
      alphaTest: 0.08,
      depthWrite: false,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    scene.add(points);
    sceneRef.current = { scene, camera, renderer, controls, points, grid, texture };

    const resize = () => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();
    let animationFrame = 0;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(render);
    };
    render();
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      points.geometry.dispose();
      material.dispose();
      texture.dispose();
      grid.geometry.dispose();
      gridMaterials.forEach((item) => item.dispose());
      renderer.dispose();
      renderer.domElement.remove();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const geometry = new THREE.BufferGeometry();
    if (cloud) {
      geometry.setAttribute('position', new THREE.BufferAttribute(cloud.positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colorMode === 'depth' ? cloud.depthColors : cloud.rgbColors, 3));
    }
    scene.points.geometry.dispose();
    scene.points.geometry = geometry;
    if (!cloud) {
      lastSceneRef.current = null;
    } else if (cloud.count > 0 && sceneKey !== lastSceneRef.current) {
      if (sceneKey !== 'demo') fitLiveCloud(scene, cloud);
      else {
        scene.grid.position.set(0, -1.09, -3);
        scene.grid.scale.setScalar(1);
        scene.camera.position.set(4.9, 3.8, 5.1);
        scene.controls.target.set(0, 0, -3.1);
        scene.controls.update();
        scene.controls.saveState();
      }
      lastSceneRef.current = sceneKey;
    }
  }, [cloud, colorMode, sceneKey]);

  return (
    <div className="point-cloud-canvas" ref={mountRef} aria-label="Интерактивное трёхмерное облако точек">
      {webGlError && <div className="webgl-error">Не удалось открыть 3D. Включите WebGL в настройках браузера.</div>}
    </div>
  );
});

export default PointCloud;
