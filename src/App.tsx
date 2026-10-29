import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, ArrowDownToLine, ArrowLeftRight, ArrowRight, Box, Camera, Check,
  ChevronRight, CircleHelp, Focus, Grid2X2, Info,
  LayoutDashboard, LoaderCircle, MousePointer2, PawPrint, Plus, Minus, Power, RefreshCw,
  RotateCcw, ScanLine, Settings2, ShieldCheck, SlidersHorizontal, Sparkles,
  TriangleAlert, Video, X,
} from 'lucide-react';
import CameraFeed from './components/CameraFeed';
import CombinedView from './components/CombinedView';
import DepthMap from './components/DepthMap';
import Guide from './components/Guide';
import HamsterFrame from './components/HamsterFrame';
import PointCloud, { type PointCloudHandle } from './components/PointCloud';
import { useCameras } from './hooks/useCameras';
import { useStereo } from './hooks/useStereo';
import { createDemoCloud } from './lib/demo';
import { downloadPly } from './lib/export';
import { describeHamsterPosition } from './lib/hamster';
import { describeSetupReason } from './lib/stitch';
import type { CameraOrder } from './lib/stereo';

type SourceMode = 'demo' | 'live';
type ViewMode = 'combined' | 'dual' | 'depth3d' | 'hamster';
type DepthSub = 'cloud' | 'map';
type ColorMode = 'depth' | 'rgb';

const defaults = { baselineCm: 6.5, horizontalFov: 60, maxDepthM: 8 };
function matchesLabel(count: number) {
  const lastTwo = count % 100;
  const word = lastTwo >= 11 && lastTwo <= 14 ? 'совпадений' : count % 10 === 1 ? 'совпадение' : count % 10 >= 2 && count % 10 <= 4 ? 'совпадения' : 'совпадений';
  return `${count} ${word}`;
}
function initialSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem('stereo-space-settings') || '{}') as Record<string, unknown>;
    const valid = (value: unknown, min: number, max: number, fallback: number) => typeof value === 'number' && value >= min && value <= max ? value : fallback;
    return {
      baselineCm: valid(stored.baselineCm, 1, 200, defaults.baselineCm),
      horizontalFov: valid(stored.horizontalFov, 20, 150, defaults.horizontalFov),
      maxDepthM: valid(stored.maxDepthM, 1, 30, defaults.maxDepthM),
    };
  } catch { return defaults; }
}

function Brand() {
  return <div className="brand"><div className="brand-mark" aria-hidden="true"><span /><span /><i /></div><div><div className="brand-name">stereo<span>space</span><b>.</b></div><div className="brand-sub">STEREO VISION STUDIO</div></div></div>;
}

function Step({ number, title, text, state }: { number: string; title: string; text: string; state: 'done' | 'active' | 'waiting' | 'warning' }) {
  return <div className={`flow-step ${state}`}><div className="step-number">{state === 'done' ? <Check size={14} strokeWidth={2.4} /> : number}</div><div><div className="step-title">{title}</div><div className="step-text">{text}</div></div></div>;
}

function ParamInput({ label, value, min, max, suffix, onValue, description }: { label: string; value: number; min: number; max: number; suffix: string; onValue: (value: number) => void; description: string }) {
  const [draft, setDraft] = useState(String(value).replace('.', ','));
  const [invalid, setInvalid] = useState(false);
  const change = (text: string) => {
    setDraft(text);
    const number = Number(text.replace(',', '.'));
    const isValid = text.trim() !== '' && Number.isFinite(number) && number >= min && number <= max;
    setInvalid(!isValid);
    if (isValid) onValue(number);
  };
  return <label className="parameter"><span className="parameter-label">{label} <span className="parameter-info" title={description}><Info size={12} /></span></span><span className={`parameter-input ${invalid ? 'invalid' : ''}`}><input type="text" inputMode="decimal" value={draft} onChange={(event) => change(event.target.value)} onBlur={() => { setDraft(String(value).replace('.', ',')); setInvalid(false); }} aria-invalid={invalid} aria-label={label} /><span>{suffix}</span></span>{invalid && <span className="input-error">От {min} до {max}</span>}</label>;
}

/** Big raw streams for connection / overlap checks — never a fake merged view. */
function RawPair({ streamA, streamB, order, skewMs }: { streamA: MediaStream | null; streamB: MediaStream | null; order: CameraOrder | null; skewMs: number }) {
  const refA = useRef<HTMLVideoElement>(null);
  const refB = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const video = refA.current;
    if (!video) return;
    video.srcObject = streamA ?? null;
    if (streamA) void video.play().catch(() => { /* Browser retries with autoplay. */ });
    return () => { video.srcObject = null; };
  }, [streamA]);
  useEffect(() => {
    const video = refB.current;
    if (!video) return;
    video.srcObject = streamB ?? null;
    if (streamB) void video.play().catch(() => { /* Browser retries with autoplay. */ });
    return () => { video.srcObject = null; };
  }, [streamB]);
  const label = (slot: 'a' | 'b') => (!order ? `КАМЕРА ${slot === 'a' ? '01' : '02'}` : ((order === 'a-left') === (slot === 'a') ? 'ЛЕВАЯ' : 'ПРАВАЯ'));
  return (
    <div className="raw-pair">
      <div className="raw-cell">
        <video ref={refA} autoPlay muted playsInline className={streamA ? 'visible' : ''} />
        <span className="raw-label">{label('a')}</span>
        {!streamA && <div className="camera-empty"><Camera size={19} strokeWidth={1.4} /><span>Нет сигнала</span></div>}
      </div>
      <div className="raw-cell">
        <video ref={refB} autoPlay muted playsInline className={streamB ? 'visible' : ''} />
        <span className="raw-label">{label('b')}</span>
        {!streamB && <div className="camera-empty"><Camera size={19} strokeWidth={1.4} /><span>Нет сигнала</span></div>}
      </div>
      {skewMs > 0 && <div className="raw-skew">Сопоставление кадров программное: разница времени кадров ≈ {Math.round(skewMs)} мс</div>}
    </div>
  );
}

function App() {
  const cameras = useCameras();
  const videoA = useRef<HTMLVideoElement>(null);
  const videoB = useRef<HTMLVideoElement>(null);
  const pointCloudRef = useRef<PointCloudHandle>(null);
  const demoCloud = useMemo(createDemoCloud, []);
  const [settings, setSettings] = useState(initialSettings);
  const [source, setSource] = useState<SourceMode>('demo');
  const [view, setView] = useState<ViewMode>('dual');
  const [depthSub, setDepthSub] = useState<DepthSub>('cloud');
  const [colorMode, setColorMode] = useState<ColorMode>('depth');
  const [gridVisible, setGridVisible] = useState(true);
  const [guideOpen, setGuideOpen] = useState(false);
  const [manualOpen, setManualOpen] = useState(false);
  const stereo = useStereo({ slots: cameras.slots, videoA, videoB, active: source === 'live', ...settings });
  const cameraCount = Number(!!cameras.slots.a) + Number(!!cameras.slots.b);
  const pairReady = cameraCount === 2;
  const cloud = source === 'demo' ? demoCloud : stereo.cloud;
  const hasLivePoints = source === 'live' && !!stereo.cloud?.count;
  const closeGuide = useCallback(() => setGuideOpen(false), []);

  useEffect(() => {
    try { localStorage.setItem('stereo-space-settings', JSON.stringify(settings)); } catch { /* Storage is optional. */ }
  }, [settings]);

  useEffect(() => {
    if (cameras.slots.a && cameras.slots.b) setSource('live');
    setManualOpen(false);
  }, [cameras.slots.a?.stream, cameras.slots.b?.stream]);

  // The combined view is the primary mode only after a trustworthy merge.
  useEffect(() => {
    if (source !== 'live') return;
    if (stereo.combineStatus === 'ready' || stereo.combineStatus === 'planar' || stereo.combineStatus === 'unreliable-depth') {
      setView((current) => (current === 'dual' ? 'combined' : current));
    } else if (stereo.combineStatus === 'insufficient' || stereo.combineStatus === 'low-overlap' || stereo.combineStatus === 'desync') {
      setView((current) => (current === 'combined' ? 'dual' : current));
    }
  }, [stereo.combineStatus, source]);

  const connect = () => { setSource('live'); void cameras.connect(); };
  const disconnect = () => { cameras.disconnect(); setSource('demo'); };
  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  const choose = (order: CameraOrder) => { setSource('live'); stereo.chooseOrder(order); setManualOpen(false); };
  const swap = () => choose(stereo.order === 'a-left' ? 'b-left' : 'a-left');
  const position = (slot: 'a' | 'b') => !stereo.order ? 'Не определена' : ((stereo.order === 'a-left') === (slot === 'a') ? 'Левая' : 'Правая');
  const statusText = stereo.status === 'automatic' ? 'Определено автоматически' : stereo.status === 'manual' ? 'Порядок задан вручную' : stereo.status === 'detecting' ? 'Ищем совпадения в кадрах' : stereo.status === 'failed' ? 'Не удалось определить стороны' : 'Ожидает две камеры';
  const orderStep = stereo.order ? (stereo.status === 'manual' ? 'Порядок выбран' : 'Стороны определены') : stereo.status === 'detecting' ? 'Идёт определение' : stereo.status === 'failed' ? 'Выберите вручную' : 'Ожидает камер';
  const combineReady = stereo.combineStatus === 'ready' || stereo.combineStatus === 'planar';
  const viewStep = source === 'demo' ? 'Демо-просмотр' : combineReady ? 'Общий вид готов' : stereo.combineStatus === 'unreliable-depth' ? 'Общий вид · ненадёжная глубина' : stereo.status === 'detecting' ? 'Настраиваем' : 'Ожидает настройку';
  const range = cloud?.near != null && cloud.far != null ? `${cloud.near.toFixed(1).replace('.', ',')}–${cloud.far.toFixed(1).replace('.', ',')} м` : '—';
  const expectedLeft = stereo.order === 'a-left' ? 'a' : 'b';
  const observation = source === 'live' && pairReady && stereo.order && stereo.observation?.leftSlot === expectedLeft ? stereo.observation : null;
  const hamster = observation?.box ?? null;
  const hamsterState = !pairReady ? 'Ожидает две камеры'
    : source === 'demo' ? 'Откройте камеры'
      : !stereo.order ? 'Ожидает порядок камер'
        : stereo.hamsterStatus === 'loading' ? 'Загружаем модель'
          : stereo.hamsterStatus === 'error' ? 'Распознавание недоступно'
            : !observation ? 'Ожидаем свежий кадр'
              : hamster ? 'Похоже, хомяк' : 'Хомяк не найден';
  const hamsterHelp = !pairReady ? 'Подключите две камеры, чтобы начать наблюдение.'
    : source === 'demo' ? 'Вернитесь к камерам: демо-сцена не анализируется.'
      : !stereo.order ? 'Дождитесь определения сторон или выберите левую камеру вручную.'
        : stereo.hamsterStatus === 'loading' ? 'Модель загружается с этого сервера и работает на устройстве.'
          : stereo.hamsterStatus === 'error' ? 'Не удалось загрузить модель или запустить WebAssembly. Проверьте файлы приложения и браузер.'
            : 'Кадр и рамка обновляются после обработки. Результат может быть ошибочным.';
  const distance = observation?.distance?.metres;
  const distanceText = distance != null ? `≈ ${distance.toFixed(distance < 1 ? 2 : 1).replace('.', ',')} м` : 'Не измерено';
  const xPercent = hamster ? Math.round((hamster.left + hamster.right) * 50) : 0;
  const yPercent = hamster ? Math.round((hamster.top + hamster.bottom) * 50) : 0;

  // Real status of the combined view — every wording maps to a measured state.
  const setupFailure = stereo.setupAssessment && !stereo.setupAssessment.ready ? describeSetupReason(stereo.setupAssessment.reason) : null;
  const statusInfo: { text: string; tone: 'idle' | 'busy' | 'ready' | 'warn' | 'demo'; detail: string } = (() => {
    if (source === 'demo') return { text: 'Демо-сцена', tone: 'demo', detail: 'Не изображение с камер. Подключите камеры для общего вида.' };
    if (!pairReady) return { text: 'Нет сигнала камер', tone: 'idle', detail: 'Подключите две разные камеры.' };
    if (stereo.profileNote === 'stale') return { text: 'Требуется повторная калибровка', tone: 'warn', detail: 'Камеры, разрешение или положение изменились с прошлой настройки. Запустите «Автонастройку».' };
    if (stereo.status === 'detecting' || stereo.combineStatus === 'setup') return { text: 'Настраиваем', tone: 'busy', detail: 'Сопоставляем общие детали двух камер. Держите сцену неподвижной.' };
    if (stereo.status === 'failed') {
      const failure = setupFailure ?? describeSetupReason('inconsistent');
      return { text: stereo.setupAssessment?.reason === 'low-overlap' ? 'Недостаточно перекрытия' : 'Настройка не завершена', tone: 'warn', detail: `${failure.title}. ${failure.advice}` };
    }
    if (stereo.combineStatus === 'desync') return { text: 'Кадры рассинхронизированы', tone: 'warn', detail: 'Потоки идут с разной задержкой, поэтому совмещение остановлено, чтобы не показывать двойные объекты. Закройте другие программы с камерами и закрепите камеры.' };
    if (stereo.combineStatus === 'low-overlap') return { text: 'Недостаточно перекрытия', tone: 'warn', detail: 'Общая область кадров мала. Разверните камеры друг к другу или сведите их ближе к общей сцене.' };
    if (stereo.combineStatus === 'insufficient' || stereo.combineStatus === 'error') return { text: 'Совмещение недоступно', tone: 'warn', detail: 'Недостаточно согласованных совпадений для геометрического совмещения. Улучшите освещение, добавьте детали в общую сцену или повторите настройку.' };
    if (stereo.combineStatus === 'unreliable-depth') return { text: 'Ненадёжная глубина', tone: 'warn', detail: 'Общий вид построен, но параллакс компенсируется только там, где смещение измерено надёжно. Усильте текстуру сцены или стабилизируйте освещение.' };
    if (stereo.combineStatus === 'planar') return { text: 'Общий вид готов', tone: 'ready', detail: 'Плоский (панорамный) режим: подходит для плоских сцен; для объёмных сцен близкие объекты могут двоиться.' };
    if (stereo.combineStatus === 'ready') return { text: 'Общий вид готов', tone: 'ready', detail: 'Кадры совмещены по оценке глубины; зона перекрытия сведена плавно.' };
    return { text: 'Настраиваем', tone: 'busy', detail: 'Готовим совмещённый кадр.' };
  })();

  const metricParts: string[] = [];
  if (source === 'live' && stereo.pairMetrics) {
    metricParts.push(`${matchesLabel(stereo.pairMetrics.matches)} совпадений`);
    metricParts.push(`перекрытие ${Math.round(stereo.pairMetrics.overlapRatio * 100)}%`);
    metricParts.push(`сдвиг по высоте ${stereo.pairMetrics.verticalOffset} px`);
    metricParts.push(`разница яркости ×${stereo.pairMetrics.brightnessRatio.toFixed(2).replace('.', ',')}`);
  }
  if (source === 'live' && stereo.perf.latencyMs != null) metricParts.push(`задержка ${stereo.perf.latencyMs} мс`);
  if (source === 'live' && stereo.perf.fps > 0.5) metricParts.push(`${stereo.perf.fps.toFixed(1).replace('.', ',')} кадр/с`);
  if (source === 'live' && stereo.perf.skewMs > 1) metricParts.push(`рассинхронность ${Math.round(stereo.perf.skewMs)} мс`);
  const qualityNote = stereo.perf.workWidth < 256 ? `Обработка снижена до ${stereo.perf.workWidth}×${stereo.perf.workHeight}` : null;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Brand />
        <div className="nav-group-label">ПРОСТРАНСТВО</div>
        <nav className="navigation" aria-label="Навигация">
          <button className="nav-item active" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}><LayoutDashboard size={18} /><span>Обзор</span><i /></button>
          <button className="nav-item" onClick={() => { setView('combined'); scrollTo('viewer-title'); }}><Video size={18} /><span>Общий вид</span></button>
          <button className="nav-item" onClick={() => { setView('hamster'); scrollTo('viewer-title'); }}><PawPrint size={18} /><span>Хомяк</span></button>
          <button className="nav-item" onClick={() => scrollTo('sources')}><Camera size={18} /><span>Источники</span></button>
          <button className="nav-item" onClick={() => scrollTo('settings')}><SlidersHorizontal size={18} /><span>Параметры</span></button>
        </nav>
        <div className="sidebar-separator" />
        <div className="nav-group-label">ПОМОЩЬ</div>
        <nav className="navigation" aria-label="Помощь"><button className="nav-item" onClick={() => setGuideOpen(true)}><CircleHelp size={18} /><span>Как это работает</span></button></nav>
        <div className="sidebar-bottom"><div className="privacy-icon"><ShieldCheck size={17} /></div><div><strong>Всё остаётся у вас</strong><p>Видео обрабатывается на устройстве</p></div></div>
        <div className="sidebar-version"><span>STEREO SPACE</span><span>V 1.1</span></div>
      </aside>

      <main className="main-area">
        <div className="topbar"><div className="breadcrumb"><span>STEREO SPACE</span><ChevronRight size={13} /><strong>Рабочее пространство</strong></div><div className="topbar-actions"><span className="local-status"><i /> Локальная обработка</span><button className="top-help" onClick={() => setGuideOpen(true)}><CircleHelp size={16} /> <span>Помощь</span></button></div></div>
        <div className="content" id="overview">
          <div className="mobile-brand"><Brand /></div>
          <section className="intro">
            <div><div className="eyebrow intro-eyebrow"><span className="accent-line" /> ВАША РАБОЧАЯ ОБЛАСТЬ</div><div className="title-row"><h1>Стереопространство</h1><span className="beta-badge">BETA</span></div><p>Две камеры. Один совмещённый видеопоток.</p></div>
            <div className="intro-actions">
              {pairReady && <button className="secondary-button auto-setup-intro" onClick={() => { setSource('live'); stereo.detectAgain(); }} disabled={cameras.busy || stereo.status === 'detecting'}><ScanLine size={16} /> {stereo.status === 'detecting' ? 'Настраиваем...' : 'Автонастройка'}</button>}
              {cameraCount === 2 ? <button className="secondary-button disconnect-button" onClick={disconnect} disabled={cameras.busy}><Power size={16} /> Отключить камеры</button> : <>{cameraCount === 1 && <button className="secondary-button stop-one-button" onClick={disconnect} disabled={cameras.busy} title="Отключить камеру" aria-label="Отключить камеру"><Power size={16} /></button>}<button className="primary-button connect-button" onClick={connect} disabled={cameras.busy}>{cameras.busy ? <LoaderCircle size={17} className="spin" /> : <Camera size={17} />} {cameras.busy ? 'Подключаем...' : cameraCount === 1 ? 'Переподключить камеры' : 'Подключить камеры'} {!cameras.busy && <ArrowRight size={16} />}</button></>}
            </div>
          </section>

          <section className="flow-bar" aria-label="Этапы настройки">
            <Step number="01" title="ПОДКЛЮЧЕНИЕ" text={cameraCount === 2 ? '2 камеры подключены' : cameraCount === 1 ? 'Подключена 1 из 2' : 'Подключите камеры'} state={cameraCount === 2 ? 'done' : cameraCount === 1 || cameras.busy ? 'active' : 'waiting'} />
            <div className="flow-divider" />
            <Step number="02" title="АВТОНАСТРОЙКА" text={stereo.status === 'detecting' ? 'Идёт настройка' : stereo.status === 'failed' ? 'Повторите настройку' : stereo.combineStatus === 'ready' || stereo.combineStatus === 'planar' ? 'Геометрия определена' : orderStep} state={combineReady ? 'done' : stereo.status === 'detecting' ? 'active' : stereo.status === 'failed' || stereo.profileNote === 'stale' ? 'warning' : 'waiting'} />
            <div className="flow-divider" />
            <Step number="03" title="ОБЩИЙ ВИД" text={viewStep} state={combineReady ? 'done' : stereo.combineStatus === 'unreliable-depth' ? 'warning' : source === 'live' && pairReady && stereo.order ? 'active' : 'waiting'} />
          </section>

          <div className="workspace-grid">
            <section className="visual-column" aria-labelledby="viewer-title">
              <div className="section-header visual-section-header"><div><div className="section-kicker">ПРЕДПРОСМОТР <span>/</span> 01</div><h2 id="viewer-title">Общий вид <span className="heading-dot">.</span></h2></div><button className="export-button" disabled={!hasLivePoints} onClick={() => stereo.cloud && downloadPly(stereo.cloud, colorMode === 'depth')} title={hasLivePoints ? 'Скачать облако точек в формате PLY' : 'Экспорт станет доступен после построения сцены с камер'}><ArrowDownToLine size={16} /> <span>Экспорт .PLY</span></button></div>

              <div className="viewer-frame">
                <div className="viewer-toolbar">
                  <div className="viewer-caption"><span className="viewer-caption-icon"><Box size={15} /></span><span>ПРОСМОТР СЦЕНЫ</span><span className="caption-divider" /><span className="viewer-subcaption">{view === 'combined' ? 'Один совмещённый поток' : view === 'dual' ? 'Исходные потоки' : view === 'depth3d' ? (depthSub === 'cloud' ? 'Облако точек' : 'Карта глубины') : 'Наблюдение за хомяком'}</span></div>
                  <div className="view-tabs" role="group" aria-label="Режим просмотра">
                    <button className={view === 'combined' ? 'selected' : ''} onClick={() => setView('combined')}>Общий вид</button>
                    <button className={view === 'dual' ? 'selected' : ''} onClick={() => setView('dual')}>Две камеры</button>
                    <button className={view === 'depth3d' ? 'selected' : ''} onClick={() => { setView('depth3d'); setGridVisible(true); }}>Глубина / 3D</button>
                    <button className={view === 'hamster' ? 'selected' : ''} onClick={() => setView('hamster')}>Хомяк</button>
                  </div>
                </div>
                <div className={`visualization ${view === 'depth3d' && depthSub === 'map' ? 'show-depth' : ''}`}>
                  {view === 'combined' && source === 'live' && stereo.combined && (stereo.combineStatus === 'ready' || stereo.combineStatus === 'planar' || stereo.combineStatus === 'unreliable-depth') ? (
                    <CombinedView combined={stereo.combined} box={stereo.boxOnCombined} showBox={!!hamster} />
                  ) : view === 'combined' && source === 'live' ? (
                    <RawPair streamA={cameras.slots.a?.stream ?? null} streamB={cameras.slots.b?.stream ?? null} order={stereo.order} skewMs={stereo.perf.skewMs} />
                  ) : view === 'dual' && source === 'live' ? (
                    <RawPair streamA={cameras.slots.a?.stream ?? null} streamB={cameras.slots.b?.stream ?? null} order={stereo.order} skewMs={stereo.perf.skewMs} />
                  ) : view === 'depth3d' ? (
                    depthSub === 'cloud' ? <PointCloud ref={pointCloudRef} cloud={cloud} colorMode={colorMode} sceneKey={source === 'demo' ? 'demo' : `${cameras.slots.a?.id ?? ''}|${cameras.slots.b?.id ?? ''}`} /> : <DepthMap cloud={cloud} />
                  ) : view === 'hamster' && observation ? (
                    <HamsterFrame observation={observation} className="hamster-main-frame" />
                  ) : view === 'hamster' ? null : (
                    <RawPair streamA={cameras.slots.a?.stream ?? null} streamB={cameras.slots.b?.stream ?? null} order={stereo.order} skewMs={stereo.perf.skewMs} />
                  )}

                  {view !== 'hamster' && view !== 'combined' && view !== 'dual' && <div className="viewport-vignette" />}
                  {view !== 'hamster' && <div className="viewport-top-left">
                    <div className={`source-pill ${source === 'demo' ? 'demo-pill' : view === 'combined' && combineReady ? 'live-pill' : 'raw-pill'}`}><span /> {source === 'demo' ? 'ДЕМО-СЦЕНА' : view === 'combined' && combineReady ? 'СОВМЕЩЁННЫЙ ПОТОК' : 'ИСХОДНЫЕ ПОТОКИ'}</div>
                    <div className="viewport-description">{source === 'demo' ? 'Пример визуализации · не с камер' : view === 'combined' && combineReady ? (stereo.combined?.mode === 'planar' ? 'Плоский режим совмещения' : 'Совмещение по оценке глубины') : 'Без совмещения'}</div>
                  </div>}
                  {view === 'hamster' && observation && <div className="hamster-frame-caption"><span /> АНАЛИЗ КАДРА ЛЕВОЙ КАМЕРЫ</div>}
                  {source === 'demo' && pairReady && view !== 'hamster' && <button className="back-live" onClick={() => setSource('live')}>К камерам <ArrowRight size={13} /></button>}
                  {view === 'depth3d' && depthSub === 'cloud' && <div className="viewport-tools"><button title="Увеличить" aria-label="Увеличить" onClick={() => pointCloudRef.current?.zoomIn()}><Plus size={16} /></button><button title="Уменьшить" aria-label="Уменьшить" onClick={() => pointCloudRef.current?.zoomOut()}><Minus size={16} /></button><div /><button title="Сбросить ракурс" aria-label="Сбросить ракурс" onClick={() => pointCloudRef.current?.reset()}><RotateCcw size={15} /></button><button className={gridVisible ? 'tool-active' : ''} title={gridVisible ? 'Скрыть сетку' : 'Показать сетку'} aria-label={gridVisible ? 'Скрыть сетку' : 'Показать сетку'} onClick={() => setGridVisible(pointCloudRef.current?.toggleGrid() ?? false)}><Grid2X2 size={15} /></button></div>}
                  {view === 'depth3d' && source === 'live' && hasLivePoints && <div className="depth-sub-switch" role="group" aria-label="Режим глубины"><button className={depthSub === 'cloud' ? 'selected' : ''} onClick={() => setDepthSub('cloud')}>Облако</button><button className={depthSub === 'map' ? 'selected' : ''} onClick={() => setDepthSub('map')}>Карта</button></div>}

                  {(view === 'combined' || view === 'dual') && source !== 'demo' && (
                    <div className={`merge-status ${statusInfo.tone}`} role="status">
                      <div className="merge-status-head">
                        {statusInfo.tone === 'busy' ? <LoaderCircle size={15} className="spin" /> : statusInfo.tone === 'ready' ? <Check size={15} /> : statusInfo.tone === 'warn' ? <TriangleAlert size={15} /> : <ScanLine size={15} />}
                        <strong>{statusInfo.text}</strong>
                        {view === 'combined' && combineReady && <span className="merge-mode-chip">{stereo.combined?.mode === 'planar' ? 'ПЛОСКИЙ РЕЖИМ' : 'ПО ГЛУБИНЕ'}</span>}
                      </div>
                      <p>{statusInfo.detail}</p>
                      {setupFailure && !combineReady && <div className="merge-advice"><strong>Что сделать:</strong> {setupFailure.advice}</div>}
                      {qualityNote && <div className="merge-advice">{qualityNote}</div>}
                    </div>
                  )}
                  {source === 'demo' && view !== 'hamster' && <div className="scene-empty"><div className="scene-empty-icon"><ScanLine size={23} /></div><h3>Подключите две камеры</h3><p>Совмещённый вид строится только из реальных кадров. Демо-сцена не подменяет камеры.</p><button onClick={connect}>Подключить камеры <ArrowRight size={14} /></button></div>}
                  {view === 'hamster' && !observation && <div className="scene-empty hamster-empty"><div className="scene-empty-icon">{stereo.hamsterStatus === 'loading' && stereo.order ? <LoaderCircle size={23} className="spin" /> : <PawPrint size={23} />}</div><h3>{hamsterState}</h3><p>{hamsterHelp}</p>{source === 'demo' && pairReady && <button onClick={() => setSource('live')}>К камерам <ArrowRight size={14} /></button>}{stereo.hamsterStatus === 'error' && pairReady && !!stereo.order && <button onClick={stereo.retryHamster}>Повторить <RefreshCw size={14} /></button>}</div>}
                  {view === 'depth3d' && source === 'live' && !!stereo.cloud?.count && stereo.cloud.count < 70 && <div className="few-points"><TriangleAlert size={14} /> Мало совпадений — проверьте расположение камер</div>}
                  {view === 'depth3d' && <div className="viewport-bottom"><div className="viewport-coordinates"><span className="crosshair">✳</span> {depthSub === 'cloud' ? 'СТЕРЕО / 3D' : 'СТЕРЕО / ГЛУБИНА'}</div><div className="depth-legend"><span>ДАЛЬШЕ</span><div /><span>БЛИЖЕ</span></div></div>}
                </div>
                <div className="viewer-footer">
                  <div className="viewer-hint"><MousePointer2 size={15} /><span>{view === 'combined' ? (stereo.combined?.mode === 'planar' ? 'Плоский режим — для плоских сцен' : 'Параллакс компенсируется по глубине') : view === 'dual' ? 'Исходные кадры без обработки' : view === 'depth3d' && depthSub === 'cloud' ? 'Перетащите, чтобы повернуть' : view === 'depth3d' ? 'Цвет показывает глубину' : 'Рамка относится к показанному кадру'}</span><span className="hint-dot">·</span><span className="hint-secondary">{view === 'combined' ? 'Нет данных — не заполняются' : view === 'dual' ? 'Для проверки подключения и перекрытия' : view === 'depth3d' && depthSub === 'cloud' ? 'Колесо — масштаб' : view === 'depth3d' ? 'Тёмные области — нет данных' : 'Анализ локально'}</span></div>
                  {view === 'depth3d' && <div className="color-switch"><span>ЦВЕТ</span><button className={colorMode === 'depth' ? 'selected' : ''} onClick={() => setColorMode('depth')}>Глубина</button><button className={colorMode === 'rgb' ? 'selected' : ''} onClick={() => setColorMode('rgb')}>RGB</button></div>}
                </div>
              </div>

              {metricParts.length > 0 && <div className="merge-metrics" aria-label="Показатели качества">{metricParts.map((part) => <span key={part}>{part}</span>)}</div>}

              <section className="hamster-summary" aria-labelledby="hamster-title">
                <div className="hamster-summary-heading">
                  <div className="hamster-summary-title"><span className="hamster-icon"><PawPrint size={18} /></span><div><span>ЛОКАЛЬНОЕ РАСПОЗНАВАНИЕ</span><h3 id="hamster-title">Наблюдение за хомяком</h3></div></div>
                  <button onClick={() => setView('hamster')}>Открыть кадр <ArrowRight size={14} /></button>
                </div>
                <div className="hamster-metrics">
                  <div><span>СТАТУС</span><strong className={hamster ? 'hamster-found' : ''}>{hamsterState}</strong><small>{hamster ? `оценка модели ${Math.round(hamster.score * 100)}%` : hamsterHelp}</small></div>
                  <div><span>ПОЛОЖЕНИЕ В КАДРЕ</span><strong>{hamster ? describeHamsterPosition(hamster) : '—'}</strong><small>{hamster ? `центр: X ${xPercent}% · Y ${yPercent}% · одна рамка на общий вид` : 'по левой камере'}</small></div>
                  <div><span>ПРИБЛ. ГЛУБИНА</span><strong>{hamster ? distanceText : '—'}</strong><small>{hamster ? distance != null ? `${matchesLabel(observation?.distance?.samples ?? 0)} в рамке` : 'недостаточно надёжных совпадений' : 'по исходной стереопаре'}</small></div>
                </div>
                <p className="hamster-disclaimer">Модель может ошибаться. Глубина приблизительна: рамка может содержать фон, а камеры не откалиброваны. Глубина считается по исходной стереопаре, а не по совмещённому кадру.</p>
                {stereo.hamsterStatus === 'error' && pairReady && !!stereo.order && <button className="hamster-retry" onClick={stereo.retryHamster}>Повторить запуск модели <RefreshCw size={13} /></button>}
              </section>

              <div className="scene-stats"><div><span>ТОЧКИ СЦЕНЫ</span><strong>{cloud ? cloud.count.toLocaleString('ru-RU') : '—'}</strong><small>{source === 'demo' ? 'в демо-сцене' : 'найдено в кадре'}</small></div><div><span>ДИАПАЗОН ГЛУБИНЫ</span><strong>{source === 'live' ? range : '—'}</strong><small>{source === 'live' && distance == null ? 'приблизительно' : 'по стереопаре'}</small></div><div><span>ИСТОЧНИК ДАННЫХ</span><strong className="stat-source"><i className={source === 'live' && pairReady ? 'on' : ''} /> {source === 'demo' ? 'Демонстрация' : pairReady ? '2 камеры' : 'Нет сигнала'}</strong><small>{source === 'demo' ? 'не с камер' : stereo.processingMs ? `обработка ~${stereo.processingMs} мс` : 'локальная обработка'}</small></div></div>
              <button className="under-view-help" onClick={() => setGuideOpen(true)}><span className="under-help-icon"><Sparkles size={17} /></span><span><strong>Впервые здесь?</strong> Посмотрите, как расположить камеры для лучшего результата.</span><ArrowRight size={17} /></button>
            </section>

            <aside className="control-column">
              <section className="control-panel sources-panel" id="sources" aria-labelledby="sources-title"><div className="panel-heading"><div className="panel-title-group"><span className="panel-icon"><Video size={17} /></span><div><div className="panel-kicker">ИСТОЧНИКИ</div><h2 id="sources-title">Камеры</h2></div></div><span className={`count-badge ${pairReady ? 'complete' : ''}`}>{cameraCount} <span>/</span> 2</span></div><p className="panel-description">Два взгляда на одну сцену</p>
                <div className="cameras-grid"><CameraFeed slot="a" camera={cameras.slots.a} otherId={cameras.slots.b?.id} devices={cameras.devices} videoRef={videoA} position={position('a')} busy={cameras.busy} observation={observation?.leftSlot === 'a' ? observation : null} onChange={cameras.changeCamera} /><CameraFeed slot="b" camera={cameras.slots.b} otherId={cameras.slots.a?.id} devices={cameras.devices} videoRef={videoB} position={position('b')} busy={cameras.busy} observation={observation?.leftSlot === 'b' ? observation : null} onChange={cameras.changeCamera} /></div>
                {cameras.error && <div className="inline-error" role="alert"><TriangleAlert size={15} /><span>{cameras.error}</span><button aria-label="Скрыть сообщение" onClick={cameras.clearError}><X size={14} /></button></div>}
                {stereo.processingError && <div className="inline-error" role="alert"><TriangleAlert size={15} /><span>{stereo.processingError}</span></div>}
                <div className="order-block"><div className="order-header"><span>АВТОНАСТРОЙКА</span><button className="refresh-devices" onClick={() => void cameras.refresh()} disabled={cameras.busy || !cameras.devices.length} title="Обновить список устройств" aria-label="Обновить список устройств"><RefreshCw size={14} /></button></div>
                  <div className="setup-checklist">
                    <div className={`setup-check ${pairReady ? 'done' : ''}`}>{pairReady ? <Check size={13} /> : <span>1</span>} Две камеры подключены</div>
                    <div className={`setup-check ${stereo.order ? 'done' : stereo.status === 'detecting' ? 'active' : ''}`}>{stereo.order ? <Check size={13} /> : <span>2</span>} Стороны и совпадения {stereo.pairMetrics ? `· ${matchesLabel(stereo.pairMetrics.matches)}` : ''}</div>
                    <div className={`setup-check ${stereo.pairMetrics && stereo.pairMetrics.overlapRatio > 0 ? (stereo.pairMetrics.overlapRatio >= 0.35 ? 'done' : 'warn') : ''}`}>{stereo.pairMetrics?.overlapRatio != null && stereo.pairMetrics.overlapRatio >= 0.35 ? <Check size={13} /> : <span>3</span>} Перекрытие кадров{stereo.pairMetrics ? ` · ${Math.round(stereo.pairMetrics.overlapRatio * 100)}%` : ''}</div>
                    <div className={`setup-check ${combineReady ? 'done' : stereo.combineStatus === 'unreliable-depth' ? 'warn' : ''}`}>{combineReady ? <Check size={13} /> : <span>4</span>} Совмещённый кадр</div>
                  </div>
                  <div className="order-status"><span className={`order-status-icon ${stereo.order ? 'success' : stereo.status === 'failed' ? 'warning' : ''}`}>{stereo.order ? <Check size={14} /> : stereo.status === 'detecting' ? <LoaderCircle size={14} className="spin" /> : stereo.status === 'failed' ? <TriangleAlert size={14} /> : <Focus size={14} />}</span><div><strong>{statusText}</strong><p>{stereo.status === 'automatic' ? `${stereo.orientation?.matches ?? '—'} общих деталей в кадре` : stereo.status === 'manual' ? 'Можно поменять стороны местами' : stereo.status === 'detecting' ? 'Держите камеры неподвижно' : stereo.status === 'failed' ? 'Проверьте общую область кадра' : 'Стороны определятся по изображению'}</p></div></div>
                  {stereo.profileNote === 'saved' && <div className="profile-note ok">Профиль сохранён для этой пары камер и режима съёмки.</div>}
                  {stereo.profileNote === 'loaded' && <div className="profile-note ok">Загружен сохранённый профиль — геометрия восстановлена без повторного замера.</div>}
                  {stereo.profileNote === 'stale' && <div className="profile-note warn">Требуется повторная калибровка: камеры, разрешение или положение изменились.</div>}
                  {pairReady && (manualOpen || stereo.status === 'failed') && !stereo.order && <div className="manual-choice"><span>КАКАЯ КАМЕРА СЛЕВА?</span><div><button onClick={() => choose('a-left')}>Камера 01</button><button onClick={() => choose('b-left')}>Камера 02</button></div></div>}
                  <div className="order-actions"><button className="auto-button" disabled={!pairReady || cameras.busy || stereo.status === 'detecting'} onClick={() => { setSource('live'); setManualOpen(false); stereo.detectAgain(); }}><ScanLine size={15} /> {stereo.status === 'detecting' ? 'Настраиваем...' : 'Автонастройка'}</button>{stereo.order && <button className="swap-button" onClick={swap} title="Поменять левую и правую камеры" aria-label="Поменять левую и правую камеры"><ArrowLeftRight size={16} /></button>}</div>
                  {pairReady && !stereo.order && stereo.status !== 'failed' && <button className="manual-link" onClick={() => setManualOpen(!manualOpen)}>{manualOpen ? 'Скрыть выбор' : 'Выбрать стороны вручную'}</button>}
                </div>
              </section>
              <section className="control-panel settings-panel" id="settings" aria-labelledby="settings-title"><div className="panel-heading"><div className="panel-title-group"><span className="panel-icon"><Settings2 size={17} /></span><div><div className="panel-kicker">НАСТРОЙКА</div><h2 id="settings-title">Геометрия пары</h2></div></div><span className="settings-tag">3D</span></div><p className="panel-description">Нужна для расчёта глубины</p><div className="parameter-grid"><ParamInput label="База камер" value={settings.baselineCm} min={1} max={200} suffix="см" description="Расстояние между центрами объективов. Без этого измерения расстояния в метрах недостоверны." onValue={(value) => setSettings((current) => ({ ...current, baselineCm: value }))} /><ParamInput label="Угол обзора" value={settings.horizontalFov} min={20} max={150} suffix="°" description="Горизонтальный угол обзора после кадрирования" onValue={(value) => setSettings((current) => ({ ...current, horizontalFov: value }))} /></div><div className="range-control"><div><label htmlFor="max-depth">Дальность сцены</label><strong>{settings.maxDepthM} м</strong></div><input id="max-depth" type="range" min="1" max="30" step="1" value={settings.maxDepthM} onChange={(event) => setSettings((current) => ({ ...current, maxDepthM: Number(event.target.value) }))} /><div className="range-labels"><span>1 м</span><span>30 м</span></div></div><div className="geometry-note\"><Info size={14} /><span>Измерьте расстояние между центрами объективов (базу) в сантиметрах — это единственное физическое измерение, которое нужно для глубины в метрах. Без него значения глубины недостоверны.</span></div></section>
              <button className="setup-tip" onClick={() => setGuideOpen(true)}><span className="tip-icon"><Focus size={17} /></span><span><strong>Нужен хороший ракурс</strong><small>Направьте камеры параллельно на одну сцену с заметными деталями.</small><em>Как настроить камеры <ArrowRight size={13} /></em></span></button>
            </aside>
          </div>
          <footer className="page-footer"><span><Activity size={14} /> СОЗДАНО ДЛЯ ОБЪЁМНОГО ВЗГЛЯДА</span><span>ДАННЫЕ ОСТАЮТСЯ НА УСТРОЙСТВЕ <ShieldCheck size={14} /></span></footer>
        </div>
      </main>
      {guideOpen && <Guide onClose={closeGuide} />}
    </div>
  );
}

export default App;
