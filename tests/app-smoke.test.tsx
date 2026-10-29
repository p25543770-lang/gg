// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';

vi.mock('three', () => ({
  Scene: class {},
  PerspectiveCamera: class {},
  WebGLRenderer: class { render() {} setSize() {} dispose() {} domElement = document.createElement('canvas'); },
  Points: class {},
  GridHelper: class {},
  CanvasTexture: class {},
  BufferGeometry: class { setAttribute() {} dispose() {} },
  BufferAttribute: class {},
  PointsMaterial: class {},
  Color: class {},
  Vector3: class { set() {} },
}));
vi.mock('three/examples/jsm/controls/OrbitControls.js', () => ({ OrbitControls: class { dispose() {} update() {} addEventListener() {} target = { set() {} }; } }));

afterEach(() => cleanup());

describe('app smoke', () => {
  it('renders the combined-view workspace with all modes and real status texts', () => {
    render(<App />);
    // Mode switcher: the three required modes plus hamster.
    expect(screen.getByRole('group', { name: 'Режим просмотра' })).toBeTruthy();
    const tabs = screen.getAllByRole('button').map((b) => b.textContent);
    expect(tabs).toContain('Общий вид');
    expect(tabs).toContain('Две камеры');
    expect(tabs).toContain('Глубина / 3D');
    expect(tabs).toContain('Хомяк');
    // Auto setup button exists (in the setup panel).
    expect(screen.getAllByText('Автонастройка').length).toBeGreaterThan(0);
    // The classic stereo-depth formula from the reference pipeline is visible.
    expect(screen.getByText('Z = f × B / d')).toBeTruthy();
    expect(screen.getAllByText(/Logitech C930e/).length).toBeGreaterThan(0);
    // Without cameras the app asks for cameras and never claims a ready merge.
    expect(screen.getAllByText('Подключите две камеры').length).toBeGreaterThan(0);
    expect(screen.queryByText('Общий вид готов')).toBeNull();
    // The demo scene is explicitly labelled as not being camera footage.
    expect(screen.getByText('Пример визуализации · не с камер')).toBeTruthy();
  });
});
