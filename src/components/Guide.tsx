import { useEffect, useRef } from 'react';
import { ArrowRight, Camera, Focus, Info, Ruler, ShieldCheck, X } from 'lucide-react';

export default function Guide({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => { window.removeEventListener('keydown', onKey); previous?.focus(); };
  }, [onClose]);

  return (
    <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="guide-modal" role="dialog" aria-modal="true" aria-labelledby="guide-title">
        <div className="guide-top">
          <div className="guide-icon"><Focus size={20} /></div>
          <button ref={closeRef} className="icon-button light-icon" onClick={onClose} aria-label="Закрыть инструкцию"><X size={19} /></button>
        </div>
        <div className="eyebrow">НЕБОЛЬШОЕ РУКОВОДСТВО</div>
        <h2 id="guide-title">Две камеры, одна сцена.</h2>
        <p className="guide-lead">Несколько простых шагов помогут получить более устойчивую стереопару.</p>
        <div className="guide-steps">
          <div className="guide-step">
            <div className="guide-step-icon"><Camera size={18} /></div>
            <div><span>01 / ПОДКЛЮЧЕНИЕ</span><h3>Подключите две разные камеры</h3><p>Разрешите доступ в браузере. Если камер больше двух, выберите нужные устройства в карточках. Откройте приложение по HTTPS или на localhost.</p></div>
          </div>
          <div className="guide-step">
            <div className="guide-step-icon"><Focus size={18} /></div>
            <div><span>02 / РАСПОЛОЖЕНИЕ</span><h3>Направьте их на один сюжет</h3><p>Закрепите камеры рядом, на одной высоте, параллельно друг другу. Не зеркальте изображение. В кадре должны быть общие, хорошо освещённые детали; неподвижная сцена подходит лучше.</p></div>
          </div>
          <div className="guide-step">
            <div className="guide-step-icon"><Ruler size={18} /></div>
            <div><span>03 / ГЕОМЕТРИЯ</span><h3>Укажите расстояние между объективами</h3><p>Измерьте базу между центрами объективов и задайте горизонтальный угол обзора. Порядок камер определится по смещению деталей; при необходимости его можно выбрать вручную.</p></div>
          </div>
        </div>
        <div className="guide-note"><Info size={17} /><p><strong>О точности.</strong> Это приблизительная реконструкция без полной калибровки объективов, геометрического выравнивания и аппаратной синхронизации. Несовпадающие углы обзора, повороты камер и движение ухудшают результат. Для точных измерений нужна полноценная стереокалибровка.</p></div>
        <div className="guide-bottom"><span><ShieldCheck size={15} /> Видео обрабатывается локально</span><button className="primary-button" onClick={onClose}>Понятно <ArrowRight size={16} /></button></div>
      </div>
    </div>
  );
}
