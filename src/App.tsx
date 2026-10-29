import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, ArrowDownToLine, ArrowLeftRight, ArrowRight, Box, Camera, Check,
  ChevronRight, CircleHelp, Focus, Grid2X2, Info,
  LayoutDashboard, LoaderCircle, MousePointer2, PawPrint, Plus, Minus, Power, RefreshCw,
  RotateCcw, ScanLine, Settings2, ShieldCheck, SlidersHorizontal, Sparkles,
  TriangleAlert, Video, X,
} from 'lucide-react';
import CameraFeed from './components/CameraFeed';
import DepthMap from './components/DepthMap';
import Guide from './components/Guide';
import HamsterFrame from './components/HamsterFrame';
import PointCloud, { type PointCloudHandle } from './components/PointCloud';
import { useCameras } from './hooks/useCameras';
import { useStereo } from './hooks/useStereo';
import { createDemoCloud } from './lib/demo';
import { downloadPly } from './lib/export';
import { describeHamsterPosition } from './lib/hamster';
import type { CameraOrder } from './lib/stereo';

type SourceMode = 'demo' | 'live';
type ViewMode = 'cloud' | 'depth' | 'hamster';
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

function App() {
  const cameras = useCameras();
  const videoA = useRef<HTMLVideoElement>(null);
  const videoB = useRef<HTMLVideoElement>(null);
  const pointCloudRef = useRef<PointCloudHandle>(null);
  const demoCloud = useMemo(createDemoCloud, []);
  const [settings, setSettings] = useState(initialSettings);
  const [source, setSource] = useState<SourceMode>('demo');
  const [view, setView] = useState<ViewMode>('cloud');
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

  const connect = () => { setSource('live'); void cameras.connect(); };
  const disconnect = () => { cameras.disconnect(); setSource('demo'); };
  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  const choose = (order: CameraOrder) => { setSource('live'); stereo.chooseOrder(order); setManualOpen(false); };
  const swap = () => choose(stereo.order === 'a-left' ? 'b-left' : 'a-left');
  const position = (slot: 'a' | 'b') => !stereo.order ? 'Не определена' : ((stereo.order === 'a-left') === (slot === 'a') ? 'Левая' : 'Правая');
  const statusText = stereo.status === 'automatic' ? 'Определено автоматически' : stereo.status === 'manual' ? 'Порядок задан вручную' : stereo.status === 'detecting' ? 'Ищем совпадения в кадрах' : stereo.status === 'failed' ? 'Не удалось определить стороны' : 'Ожидает две камеры';
  const orderStep = stereo.order ? (stereo.status === 'manual' ? 'Порядок выбран' : 'Стороны определены') : stereo.status === 'detecting' ? 'Идёт определение' : stereo.status === 'failed' ? 'Выберите вручную' : 'Ожидает камер';
  const reconstructionStep = source === 'demo' ? 'Демо-просмотр' : hasLivePoints ? 'Сцена построена' : pairReady && stereo.order ? 'Построение сцены' : 'Ожидает стереопару';
  const range = cloud?.near != null && cloud.far != null ? `${cloud.near.toFixed(1).replace('.', ',')}–${cloud.far.toFixed(1).replace('.', ',')} м` : '—';
  const needSceneMessage = source === 'live' && (!pairReady || !stereo.order || !stereo.cloud?.count);
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Brand />
        <div className="nav-group-label">ПРОСТРАНСТВО</div>
        <nav className="navigation" aria-label="Навигация">
          <button className="nav-item active" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}><LayoutDashboard size={18} /><span>Обзор</span><i /></button>
          <button className="nav-item" onClick={() => { setView('hamster'); scrollTo('viewer-title'); }}><PawPrint size={18} /><span>Хомяк</span></button>
          <button className="nav-item" onClick={() => scrollTo('sources')}><Video size={18} /><span>Источники</span></button>
          <button className="nav-item" onClick={() => scrollTo('settings')}><SlidersHorizontal size={18} /><span>Параметры</span></button>
        </nav>
        <div className="sidebar-separator" />
        <div className="nav-group-label">ПОМОЩЬ</div>
        <nav className="navigation" aria-label="Помощь"><button className="nav-item" onClick={() => setGuideOpen(true)}><CircleHelp size={18} /><span>Как это работает</span></button></nav>
        <div className="sidebar-bottom"><div className="privacy-icon"><ShieldCheck size={17} /></div><div><strong>Всё остаётся у вас</strong><p>Видео обрабатывается на устройстве</p></div></div>
        <div className="sidebar-version"><span>STEREO SPACE</span><span>V 1.0</span></div>
      </aside>

      <main className="main-area">
        <div className="topbar"><div className="breadcrumb"><span>STEREO SPACE</span><ChevronRight size={13} /><strong>Рабочее пространство</strong></div><div className="topbar-actions"><span className="local-status"><i /> Локальная обработка</span><button className="top-help" onClick={() => setGuideOpen(true)}><CircleHelp size={16} /> <span>Помощь</span></button></div></div>
        <div className="content" id="overview">
          <div className="mobile-brand"><Brand /></div>
          <section className="intro">
            <div><div className="eyebrow intro-eyebrow"><span className="accent-line" /> ВАША РАБОЧАЯ ОБЛАСТЬ</div><div className="title-row"><h1>Стереопространство</h1><span className="beta-badge">BETA</span></div><p>Две камеры. Один новый взгляд на пространство.</p></div>
            <div className="intro-actions">{cameraCount === 2 ? <button className="secondary-button disconnect-button" onClick={disconnect} disabled={cameras.busy}><Power size={16} /> Отключить камеры</button> : <>{cameraCount === 1 && <button className="secondary-button stop-one-button" onClick={disconnect} disabled={cameras.busy} title="Отключить камеру" aria-label="Отключить камеру"><Power size={16} /></button>}<button className="primary-button connect-button" onClick={connect} disabled={cameras.busy}>{cameras.busy ? <LoaderCircle size={17} className="spin" /> : <Camera size={17} />} {cameras.busy ? 'Подключаем...' : cameraCount === 1 ? 'Переподключить камеры' : 'Подключить камеры'} {!cameras.busy && <ArrowRight size={16} />}</button></>}</div>
          </section>

          <section className="flow-bar" aria-label="Этапы настройки">
            <Step number="01" title="ПОДКЛЮЧЕНИЕ" text={cameraCount === 2 ? '2 камеры подключены' : cameraCount === 1 ? 'Подключена 1 из 2' : 'Подключите камеры'} state={cameraCount === 2 ? 'done' : cameraCount === 1 || cameras.busy ? 'active' : 'waiting'} />
            <div className="flow-divider" />
            <Step number="02" title="ПОРЯДОК КАМЕР" text={orderStep} state={stereo.order ? 'done' : stereo.status === 'detecting' ? 'active' : stereo.status === 'failed' ? 'warning' : 'waiting'} />
            <div className="flow-divider" />
            <Step number="03" title="РЕКОНСТРУКЦИЯ" text={reconstructionStep} state={hasLivePoints ? 'done' : source === 'demo' || (pairReady && !!stereo.order) ? 'active' : 'waiting'} />
          </section>

          <div className="workspace-grid">
            <section className="visual-column" aria-labelledby="viewer-title">
              <div className="section-header visual-section-header"><div><div className="section-kicker">ПРЕДПРОСМОТР <span>/</span> 01</div><h2 id="viewer-title">3D-пространство <span className="heading-dot">.</span></h2></div><button className="export-button" disabled={!hasLivePoints} onClick={() => stereo.cloud && downloadPly(stereo.cloud, colorMode === 'depth')} title={hasLivePoints ? 'Скачать облако точек в формате PLY' : 'Экспорт станет доступен после построения сцены с камер'}><ArrowDownToLine size={16} /> <span>Экспорт .PLY</span></button></div>

              <div className="viewer-frame">
                <div className="viewer-toolbar">
                  <div className="viewer-caption"><span className="viewer-caption-icon"><Box size={15} /></span><span>ПРОСМОТР СЦЕНЫ</span><span className="caption-divider" /><span className="viewer-subcaption">{view === 'cloud' ? 'Облако точек' : view === 'depth' ? 'Карта глубины' : 'Наблюдение за хомяком'}</span></div>
                  <div className="view-tabs" role="group" aria-label="Режим просмотра">
                    <button className={view === 'cloud' ? 'selected' : ''} onClick={() => { setView('cloud'); setGridVisible(true); }}>3D</button>
                    <button className={view === 'depth' ? 'selected' : ''} onClick={() => setView('depth')}>Глубина</button>
                    <button className={view === 'hamster' ? 'selected' : ''} onClick={() => setView('hamster')}>Хомяк</button>
                  </div>
                </div>
                <div className={`visualization ${view === 'depth' ? 'show-depth' : ''}`}>
                  {view === 'cloud' ? <PointCloud ref={pointCloudRef} cloud={cloud} colorMode={colorMode} sceneKey={source === 'demo' ? 'demo' : `${cameras.slots.a?.id ?? ''}|${cameras.slots.b?.id ?? ''}`} /> : view === 'depth' ? <DepthMap cloud={cloud} /> : observation && <HamsterFrame observation={observation} className="hamster-main-frame" />}
                  {view !== 'hamster' && <div className="viewport-vignette" />}
                  {view !== 'hamster' && <div className="viewport-top-left"><div className={`source-pill ${source === 'demo' ? 'demo-pill' : 'live-pill'}`}><span /> {source === 'demo' ? 'ДЕМО-СЦЕНА' : 'С КАМЕР'}</div><div className="viewport-description">{source === 'demo' ? 'Пример визуализации' : 'Стереореконструкция'}</div></div>}
                  {view === 'hamster' && observation && <div className="hamster-frame-caption"><span /> АНАЛИЗ КАДРА ЛЕВОЙ КАМЕРЫ</div>}
                  {source === 'demo' && pairReady && view !== 'hamster' && <button className="back-live" onClick={() => setSource('live')}>К камерам <ArrowRight size={13} /></button>}
                  {view === 'cloud' && <div className="viewport-tools"><button title="Увеличить" aria-label="Увеличить" onClick={() => pointCloudRef.current?.zoomIn()}><Plus size={16} /></button><button title="Уменьшить" aria-label="Уменьшить" onClick={() => pointCloudRef.current?.zoomOut()}><Minus size={16} /></button><div /><button title="Сбросить ракурс" aria-label="Сбросить ракурс" onClick={() => pointCloudRef.current?.reset()}><RotateCcw size={15} /></button><button className={gridVisible ? 'tool-active' : ''} title={gridVisible ? 'Скрыть сетку' : 'Показать сетку'} aria-label={gridVisible ? 'Скрыть сетку' : 'Показать сетку'} onClick={() => setGridVisible(pointCloudRef.current?.toggleGrid() ?? false)}><Grid2X2 size={15} /></button></div>}
                  {view !== 'hamster' && needSceneMessage && <div className="scene-empty"><div className="scene-empty-icon">{stereo.status === 'detecting' ? <LoaderCircle size={23} className="spin" /> : <ScanLine size={23} />}</div><h3>{!pairReady ? 'Подключите две камеры' : stereo.status === 'detecting' ? 'Определяем стороны камер' : stereo.status === 'failed' ? 'Не нашли достаточно совпадений' : stereo.order ? 'Ожидаем данные сцены' : 'Выберите порядок камер'}</h3><p>{!pairReady ? 'Объедините два видеопотока, чтобы увидеть пространство в 3D.' : stereo.status === 'detecting' ? 'Сопоставляем общие детали на изображениях. Это займёт несколько секунд.' : stereo.status === 'failed' ? 'Направьте камеры на одну сцену или выберите левую камеру вручную.' : 'Проверьте, что обе камеры смотрят на освещённую сцену с деталями.'}</p>{!pairReady && <button onClick={() => setSource('demo')}>Посмотреть демо <ArrowRight size={14} /></button>}</div>}
                  {view === 'hamster' && !observation && <div className="scene-empty hamster-empty"><div className="scene-empty-icon">{stereo.hamsterStatus === 'loading' && stereo.order ? <LoaderCircle size={23} className="spin" /> : <PawPrint size={23} />}</div><h3>{hamsterState}</h3><p>{hamsterHelp}</p>{source === 'demo' && pairReady && <button onClick={() => setSource('live')}>К камерам <ArrowRight size={14} /></button>}{stereo.hamsterStatus === 'error' && pairReady && !!stereo.order && <button onClick={stereo.retryHamster}>Повторить <RefreshCw size={14} /></button>}</div>}
                  {view !== 'hamster' && source === 'live' && !!stereo.cloud?.count && stereo.cloud.count < 70 && <div className="few-points"><TriangleAlert size={14} /> Мало совпадений — проверьте расположение камер</div>}
                  {view !== 'hamster' && <div className="viewport-bottom"><div className="viewport-coordinates"><span className="crosshair">✳</span> {view === 'cloud' ? 'СТЕРЕО / 3D' : 'СТЕРЕО / ГЛУБИНА'}</div><div className="depth-legend"><span>ДАЛЬШЕ</span><div /><span>БЛИЖЕ</span></div></div>}
                </div>
                <div className="viewer-footer">
                  <div className="viewer-hint"><MousePointer2 size={15} /><span>{view === 'cloud' ? 'Перетащите, чтобы повернуть' : view === 'depth' ? 'Цвет показывает глубину' : 'Рамка относится к показанному кадру'}</span><span className="hint-dot">·</span><span className="hint-secondary">{view === 'cloud' ? 'Колесо — масштаб' : view === 'depth' ? 'Тёмные области — нет данных' : 'Анализ локально'}</span></div>
                  {view !== 'hamster' && <div className="color-switch"><span>ЦВЕТ</span><button className={colorMode === 'depth' ? 'selected' : ''} onClick={() => setColorMode('depth')}>Глубина</button><button className={colorMode === 'rgb' ? 'selected' : ''} onClick={() => setColorMode('rgb')}>RGB</button></div>}
                </div>
              </div>

              <section className="hamster-summary" aria-labelledby="hamster-title">
                <div className="hamster-summary-heading">
                  <div className="hamster-summary-title"><span className="hamster-icon"><PawPrint size={18} /></span><div><span>ЛОКАЛЬНОЕ РАСПОЗНАВАНИЕ</span><h3 id="hamster-title">Наблюдение за хомяком</h3></div></div>
                  <button onClick={() => setView('hamster')}>Открыть кадр <ArrowRight size={14} /></button>
                </div>
                <div className="hamster-metrics">
                  <div><span>СТАТУС</span><strong className={hamster ? 'hamster-found' : ''}>{hamsterState}</strong><small>{hamster ? `оценка модели ${Math.round(hamster.score * 100)}%` : hamsterHelp}</small></div>
                  <div><span>ПОЛОЖЕНИЕ В КАДРЕ</span><strong>{hamster ? describeHamsterPosition(hamster) : '—'}</strong><small>{hamster ? `центр: X ${xPercent}% · Y ${yPercent}%` : 'по левой камере'}</small></div>
                  <div><span>ПРИБЛ. ГЛУБИНА</span><strong>{hamster ? distanceText : '—'}</strong><small>{hamster ? distance != null ? `${matchesLabel(observation?.distance?.samples ?? 0)} в рамке` : 'недостаточно надёжных совпадений' : 'вдоль оси камеры'}</small></div>
                </div>
                <p className="hamster-disclaimer">Модель может ошибаться. Глубина приблизительна: рамка может содержать фон, а камеры не откалиброваны.</p>
                {stereo.hamsterStatus === 'error' && pairReady && !!stereo.order && <button className="hamster-retry" onClick={stereo.retryHamster}>Повторить запуск модели <RefreshCw size={13} /></button>}
              </section>

              <div className="scene-stats"><div><span>ТОЧКИ СЦЕНЫ</span><strong>{cloud ? cloud.count.toLocaleString('ru-RU') : '—'}</strong><small>{source === 'demo' ? 'в демо-сцене' : 'найдено в кадре'}</small></div><div><span>ДИАПАЗОН ГЛУБИНЫ</span><strong>{source === 'live' ? range : '—'}</strong><small>приблизительно</small></div><div><span>ИСТОЧНИК ДАННЫХ</span><strong className="stat-source"><i className={source === 'live' && pairReady ? 'on' : ''} /> {source === 'demo' ? 'Демонстрация' : pairReady ? '2 камеры' : 'Нет сигнала'}</strong><small>{source === 'demo' ? 'не с камер' : stereo.processingMs ? `обработка ~${stereo.processingMs} мс` : 'локальная обработка'}</small></div></div>
              <button className="under-view-help" onClick={() => setGuideOpen(true)}><span className="under-help-icon"><Sparkles size={17} /></span><span><strong>Впервые здесь?</strong> Посмотрите, как расположить камеры для лучшего результата.</span><ArrowRight size={17} /></button>
            </section>

            <aside className="control-column">
              <section className="control-panel sources-panel" id="sources" aria-labelledby="sources-title"><div className="panel-heading"><div className="panel-title-group"><span className="panel-icon"><Video size={17} /></span><div><div className="panel-kicker">ИСТОЧНИКИ</div><h2 id="sources-title">Камеры</h2></div></div><span className={`count-badge ${pairReady ? 'complete' : ''}`}>{cameraCount} <span>/</span> 2</span></div><p className="panel-description">Два взгляда на одну сцену</p>
                <div className="cameras-grid"><CameraFeed slot="a" camera={cameras.slots.a} otherId={cameras.slots.b?.id} devices={cameras.devices} videoRef={videoA} position={position('a')} busy={cameras.busy} observation={observation?.leftSlot === 'a' ? observation : null} onChange={cameras.changeCamera} /><CameraFeed slot="b" camera={cameras.slots.b} otherId={cameras.slots.a?.id} devices={cameras.devices} videoRef={videoB} position={position('b')} busy={cameras.busy} observation={observation?.leftSlot === 'b' ? observation : null} onChange={cameras.changeCamera} /></div>
                {cameras.error && <div className="inline-error" role="alert"><TriangleAlert size={15} /><span>{cameras.error}</span><button aria-label="Скрыть сообщение" onClick={cameras.clearError}><X size={14} /></button></div>}
                {stereo.processingError && <div className="inline-error" role="alert"><TriangleAlert size={15} /><span>{stereo.processingError}</span></div>}
                <div className="order-block"><div className="order-header"><span>РАСПОЛОЖЕНИЕ КАМЕР</span><button className="refresh-devices" onClick={() => void cameras.refresh()} disabled={cameras.busy || !cameras.devices.length} title="Обновить список устройств" aria-label="Обновить список устройств"><RefreshCw size={14} /></button></div><div className="order-status"><span className={`order-status-icon ${stereo.order ? 'success' : stereo.status === 'failed' ? 'warning' : ''}`}>{stereo.order ? <Check size={14} /> : stereo.status === 'detecting' ? <LoaderCircle size={14} className="spin" /> : stereo.status === 'failed' ? <TriangleAlert size={14} /> : <Focus size={14} />}</span><div><strong>{statusText}</strong><p>{stereo.status === 'automatic' ? `${stereo.orientation?.matches ?? '—'} общих деталей в кадре` : stereo.status === 'manual' ? 'Можно поменять стороны местами' : stereo.status === 'detecting' ? 'Держите камеры неподвижно' : stereo.status === 'failed' ? 'Проверьте общую область кадра' : 'Стороны определятся по изображению'}</p></div></div>
                  {pairReady && (manualOpen || stereo.status === 'failed') && !stereo.order && <div className="manual-choice"><span>КАКАЯ КАМЕРА СЛЕВА?</span><div><button onClick={() => choose('a-left')}>Камера 01</button><button onClick={() => choose('b-left')}>Камера 02</button></div></div>}
                  <div className="order-actions"><button className="auto-button" disabled={!pairReady || cameras.busy || stereo.status === 'detecting'} onClick={() => { setSource('live'); setManualOpen(false); stereo.detectAgain(); }}><ScanLine size={15} /> {stereo.status === 'detecting' ? 'Определяем...' : stereo.order || stereo.status === 'failed' ? 'Определить заново' : 'Определить автоматически'}</button>{stereo.order && <button className="swap-button" onClick={swap} title="Поменять левую и правую камеры" aria-label="Поменять левую и правую камеры"><ArrowLeftRight size={16} /></button>}</div>
                  {pairReady && !stereo.order && stereo.status !== 'failed' && <button className="manual-link" onClick={() => setManualOpen(!manualOpen)}>{manualOpen ? 'Скрыть выбор' : 'Выбрать стороны вручную'}</button>}
                </div>
              </section>
              <section className="control-panel settings-panel" id="settings" aria-labelledby="settings-title"><div className="panel-heading"><div className="panel-title-group"><span className="panel-icon"><Settings2 size={17} /></span><div><div className="panel-kicker">НАСТРОЙКА</div><h2 id="settings-title">Геометрия пары</h2></div></div><span className="settings-tag">3D</span></div><p className="panel-description">Нужна для расчёта глубины</p><div className="parameter-grid"><ParamInput label="База камер" value={settings.baselineCm} min={1} max={200} suffix="см" description="Расстояние между центрами объективов" onValue={(value) => setSettings((current) => ({ ...current, baselineCm: value }))} /><ParamInput label="Угол обзора" value={settings.horizontalFov} min={20} max={150} suffix="°" description="Горизонтальный угол обзора после кадрирования" onValue={(value) => setSettings((current) => ({ ...current, horizontalFov: value }))} /></div><div className="range-control"><div><label htmlFor="max-depth">Дальность сцены</label><strong>{settings.maxDepthM} м</strong></div><input id="max-depth" type="range" min="1" max="30" step="1" value={settings.maxDepthM} onChange={(event) => setSettings((current) => ({ ...current, maxDepthM: Number(event.target.value) }))} /><div className="range-labels"><span>1 м</span><span>30 м</span></div></div><div className="geometry-note"><Info size={14} /><span>Измерьте расстояние между центрами объективов. Глубина рассчитывается приблизительно.</span></div></section>
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
