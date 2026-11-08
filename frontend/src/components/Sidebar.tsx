import { BookOpen, FlaskConical, Layers, Search, ShieldCheck } from 'lucide-react';
import type { Tab } from '../api';

const NAV = [
  { name: 'Search', icon: Search, sub: 'Explore your knowledge' },
  { name: 'Library', icon: BookOpen, sub: 'Documents & figures' },
  { name: 'Review', icon: ShieldCheck, sub: 'Human in the loop' },
  { name: 'Experiments', icon: FlaskConical, sub: 'Measure what matters' },
] as const;

export default function Sidebar({ tab, onSelect }: { tab: Tab; onSelect: (tab: Tab) => void }) {
  return (
    <aside className="sidebar">
      <a aria-label="Atlas home" className="brand" href="#" onClick={event => { event.preventDefault(); onSelect('Search'); }}>
        <span className="brand-mark"><Layers size={23} /></span>
        <span>atlas<small>DOCUMENT INTELLIGENCE</small></span>
      </a>
      <div className="workspace-label">WORKSPACE <span>LOCAL</span></div>
      <nav aria-label="Main navigation">
        {NAV.map(({ name, icon: Icon, sub }) => (
          <button key={name} aria-label={name} title={name} className={`nav-item ${tab === name ? 'active' : ''}`}
            onClick={() => onSelect(name)}>
            <Icon size={20} />
            <span>{name}<small>{sub}</small></span>
            {tab === name && <i />}
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="model-status"><i /> FROZEN EMBEDDINGS</div>
        <p>Sentence-BERT + CLIP<br />Your knowledge, connected.</p>
        <div className="profile">
          <span>MR</span>
          <div>Research workspace<small>Multimodal Retrieval Engine</small></div>
        </div>
      </div>
    </aside>
  );
}
