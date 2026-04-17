import { useEffect, useMemo, useState, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './DdosPage.css';

/* ─── Code snippets shown in the "Integrate" section ─────────────────── */
const codeByTab = {
  script: `<!-- paste before </body> -->
<script
  src="http://localhost:5000/api/ddos/sdk.js"
  data-api-url="http://localhost:5000"
  data-api-key="sk_live_a8x2kp9mnd3q"
  data-site-id="site_001">
</script>

<!-- listen for results -->
<script>
  document.addEventListener('zerodayDdosResult', function (e) {
    console.log(e.detail);
    // e.detail = { classification, confidence, action, blocked }
    if (e.detail.blocked) {
      console.warn('Potential DDoS detected!');
    }
  });
</script>`,

  npm: `// Server-side integration (Node.js / Express middleware)
// Capture real network flow features and call the API

const axios = require('axios');

async function detectDDoS(flowFeatures) {
  const res = await axios.post(
    'http://localhost:5000/api/ddos/detect',
    flowFeatures,
    {
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'sk_live_a8x2kp9mnd3q'
      }
    }
  );
  return res.data;
}

// Example: check a request
const result = await detectDDoS({
  duration: 0.005,
  packets_rate: 8000,
  bytes_rate: 2000000,
  syn_flag_counts: 300,
  ack_flag_counts: 2,
  // ... all 33 features
});
console.log(result);
// { classification: "attack", confidence: 0.97,
//   action: "block", blocked: true }`,

  api: `POST http://localhost:5000/api/ddos/detect

Headers:
  Content-Type: application/json
  X-API-Key: sk_live_a8x2kp9mnd3q

Body (33 flow features):
{
  "duration": 0.005,
  "packet_IAT_std": 0.0002,
  "bytes_rate": 2000000,
  "packets_rate": 8000,
  "syn_flag_counts": 300,
  "ack_flag_counts": 2,
  "bwd_packets_count": 0,
  "syn_to_ack_ratio": 150,
  "iat_uniformity": 0.95,
  "fwd_bwd_rate_ratio": 500,
  ... (see feature_names.json for all 33)
}

Response:
{
  "session_id": "a1b2c3d4",
  "classification": "attack",
  "confidence": 0.97,
  "action": "block",
  "blocked": true,
  "attack_probability": 0.97
}`,
};

/* ─── Helpers ────────────────────────────────────────────────────────── */

function timeAgo(isoString) {
  if (!isoString) return '';
  const diff = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000);
  if (diff < 5) return 'just now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function classFromDetection(classification, conf) {
  if (classification === 'attack') return 'malicious';
  const c = parseFloat(conf) || 0;
  if (c < 70) return 'suspicious';
  return 'legit';
}

function formatRps(snap) {
  if (!snap) return '—';
  const rate = snap.packets_rate || 0;
  if (rate >= 1000) return `${(rate / 1000).toFixed(1)}k rps`;
  return `${Math.round(rate)} rps`;
}

/* ─── Live Detection Stats Card ──────────────────────────────────────── */

function LiveDetectionCard() {
  const [stats, setStats] = useState({ attacks: 0, benign: 0, total: 0 });

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch('/api/ddos/sessions?limit=100');
        if (!res.ok) return;
        const data = await res.json();
        const attacks = data.filter((s) => s.classification === 'attack').length;
        const benign = data.filter((s) => s.classification === 'benign').length;
        setStats({ attacks, benign, total: data.length });
      } catch { /* ignore */ }
    };
    poll();
    const t = setInterval(poll, 2500);
    return () => clearInterval(t);
  }, []);

  const rate = stats.total > 0 ? ((stats.attacks / stats.total) * 100).toFixed(1) : '0.0';

  return (
    <div className="card accuracy-card">
      <div className="accuracy-header">
        <span className="accuracy-title">DETECTION STATS</span>
        <span className="accuracy-model">XGBoost DDoS Engine</span>
      </div>

      <div className="detection-ring-row">
        <svg viewBox="0 0 80 80" className="detection-ring">
          <circle cx="40" cy="40" r="34" fill="none" stroke="#1e1e2e" strokeWidth="6" />
          <circle cx="40" cy="40" r="34" fill="none" stroke="#ff5555" strokeWidth="6"
            strokeDasharray={`${stats.total > 0 ? (stats.attacks / stats.total) * 213.6 : 0} 213.6`}
            strokeLinecap="round" transform="rotate(-90 40 40)" />
          <text x="40" y="38" textAnchor="middle" fill="#e2e2e2" fontSize="14" fontFamily="Courier New" fontWeight="700">{rate}%</text>
          <text x="40" y="50" textAnchor="middle" fill="#6b7280" fontSize="7" fontFamily="Courier New">BLOCKED</text>
        </svg>
      </div>

      <div className="detection-stats-grid">
        <div className="detection-stat">
          <span className="detection-stat-value" style={{ color: '#ff5555' }}>{stats.attacks}</span>
          <span className="detection-stat-label">Attacks Blocked</span>
        </div>
        <div className="detection-stat">
          <span className="detection-stat-value" style={{ color: '#00d4aa' }}>{stats.benign}</span>
          <span className="detection-stat-label">Benign Allowed</span>
        </div>
        <div className="detection-stat">
          <span className="detection-stat-value" style={{ color: '#4a9eff' }}>{stats.total}</span>
          <span className="detection-stat-label">Total Flows</span>
        </div>
      </div>
    </div>
  );
}

/* ─── Component ──────────────────────────────────────────────────────── */

function DdosPage({ onSignOut }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('script');
  const [copied, setCopied] = useState(false);
  const [ddosEnabled, setDdosEnabled] = useState(true);
  const [feedData, setFeedData] = useState([]);

  const navigate = useNavigate();
  const location = useLocation();

  const dotColor = useMemo(
    () => ({
      legit: '#00d4aa',
      suspicious: '#f59e0b',
      malicious: '#ff5555',
    }),
    [],
  );

  /* ── Poll backend for live sessions ──────────────────────────────── */
  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch('/api/ddos/sessions?limit=5');
      if (!res.ok) return;
      const data = await res.json();
      setFeedData(
        data.map((s) => ({
          ip: s.ip || '0.0.0.0',
          cls: classFromDetection(s.classification, s.confidence),
          label: s.classification?.toUpperCase() || 'UNKNOWN',
          rps: formatRps(s.features_snap),
          time: timeAgo(s.timestamp),
        })),
      );
    } catch {
      /* backend not reachable — keep existing data */
    }
  }, []);

  useEffect(() => {
    fetchSessions();
    const timer = setInterval(fetchSessions, 2500);
    return () => clearInterval(timer);
  }, [fetchSessions]);

  /* ── Copy snippet ────────────────────────────────────────────────── */
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(codeByTab[activeTab]);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="dashboard-page ddos-page">
      <div className="app">
        <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
          <div className="sb-toggle" onClick={() => setSidebarOpen((prev) => !prev)}>
            <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <line x1="2" y1="4" x2="14" y2="4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="square" />
              <line x1="2" y1="8" x2="14" y2="8" stroke="currentColor" strokeWidth="1.3" strokeLinecap="square" />
              <line x1="2" y1="12" x2="14" y2="12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="square" />
            </svg>
            <span className="sb-wordmark">ZERO<span>DAY</span></span>
          </div>

          <div className={`nav-item ${location.pathname === '/dashboard' ? 'active' : ''}`} onClick={() => navigate('/dashboard')} data-tooltip="Dashboard">
            <svg viewBox="0 0 16 16" fill="none">
              <rect x="1" y="1" width="6" height="6" stroke="currentColor" strokeWidth="1.1" />
              <rect x="9" y="1" width="6" height="6" stroke="currentColor" strokeWidth="1.1" />
              <rect x="1" y="9" width="6" height="6" stroke="currentColor" strokeWidth="1.1" />
              <rect x="9" y="9" width="6" height="6" stroke="currentColor" strokeWidth="1.1" />
            </svg>
            <span className="nav-label">Dashboard</span>
          </div>

          <div className={`nav-item ${location.pathname === '/ddos' ? 'active' : ''}`} onClick={() => navigate('/ddos')} data-tooltip="DDoS">
            <svg viewBox="0 0 16 16" fill="none">
              <path d="M8 1L2 4.5v4.5c0 3 2.5 5.5 6 5.5s6-2.5 6-5.5V4.5L8 1z" stroke="currentColor" strokeWidth="1.1" />
            </svg>
            <span className="nav-label">DDoS</span>
          </div>

          <div className={`nav-item ${location.pathname === '/captcha' ? 'active' : ''}`} onClick={() => navigate('/captcha')} data-tooltip="CAPTCHA">
            <svg viewBox="0 0 16 16" fill="none">
              <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.1" />
              <path d="M5.5 8l2 2 3-3.5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
            </svg>
            <span className="nav-label">CAPTCHA</span>
          </div>

          <div className="sb-divider" />

          <div className={`nav-item ${location.pathname === '/settings' ? 'active' : ''}`} onClick={() => navigate('/settings')} data-tooltip="Settings">
            <svg viewBox="0 0 16 16" fill="none">
              <circle cx="8" cy="8" r="2.5" stroke="currentColor" strokeWidth="1.1" />
              <path
                d="M8 1v2M8 13v2M1 8h2M13 8h2 M3.2 3.2l1.4 1.4M11.4 11.4l1.4 1.4 M12.8 3.2l-1.4 1.4M4.6 11.4l-1.4 1.4"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinecap="round"
              />
            </svg>
            <span className="nav-label">Settings</span>
          </div>

          <div className="sb-bottom">
            <div className="sb-user" onClick={onSignOut}>
              <div className="sb-avatar">TK</div>
              <span className="sb-username">Tanya · Logout</span>
            </div>
          </div>
        </aside>

        <div className="main">
          <div className="topbar">
            <div className="tb-left">
              <span className="live-dot" />
              <span className="tb-path">~/</span>
              <span className="tb-page">ddos</span>
              <span className="tb-sep">·</span>
              <span className="tb-site">myshop.com</span>
            </div>
            <div className="tb-right">
              <span className="chip chip-alert">HIGH RISK</span>
              <span className="chip chip-green">PROTECTED</span>
              <span className="chip chip-neutral">sk_live_a8x2...</span>
              <div className="sb-avatar" onClick={onSignOut}>TK</div>
            </div>
          </div>

          <div className="page">
            {/* ── Engine controls ──────────────────────────────────── */}
            <div>
              <p className="sec-label">Engine controls</p>

              <div className="card">
                <div className="toggle-row">
                  <div className="t-left">
                    <span className="t-title">DDoS Detection Engine</span>
                    <span className="t-desc">
                      Analyses every incoming request using the XGBoost
                      classifier — detects volumetric and application-layer
                      attacks in real time using 33 network flow features.
                    </span>
                  </div>
                  <div className={`sw-wrap ${ddosEnabled ? 'on' : ''}`} onClick={() => setDdosEnabled((prev) => !prev)}>
                    <div className="sw-track" />
                    <div className="sw-thumb" />
                    <span className="sw-on-label">ON</span>
                    <span className="sw-off-label">OFF</span>
                  </div>
                </div>
              </div>
            </div>

            {/* ── Code snippets ────────────────────────────────────── */}
            <div>
              <p className="sec-label">Integrate into your site</p>

              <div className="code-card">
                <div className="tab-bar">
                  <div className="tabs">
                    <div className={`tab ${activeTab === 'script' ? 'active' : ''}`} onClick={() => setActiveTab('script')}>Script Tag</div>
                    <div className={`tab ${activeTab === 'npm' ? 'active' : ''}`} onClick={() => setActiveTab('npm')}>Server-side</div>
                    <div className={`tab ${activeTab === 'api' ? 'active' : ''}`} onClick={() => setActiveTab('api')}>REST API</div>
                  </div>
                  <button className={`copy-btn ${copied ? 'copied' : ''}`} onClick={handleCopy}>
                    {copied ? 'COPIED' : 'COPY'}
                  </button>
                </div>

                <div className="code-block show">
                  <pre>{codeByTab[activeTab]}</pre>
                </div>
              </div>
            </div>

            {/* ── Live request feed + Model accuracy ────────────── */}
            <div>
              <div className="feed-header">
                <p className="sec-label">Live request feed</p>
                <div className="live-indicator">
                  <div className="live-dot" />
                  LIVE
                </div>
              </div>

              <div className="feed-accuracy-row">
                <div className="card">
                  {feedData.length === 0 && (
                    <div className="feed-empty">No detections yet — waiting for API calls …</div>
                  )}
                  {feedData.map((f, idx) => (
                    <div className="feed-item" key={`${f.ip}-${idx}`}>
                      <div className="feed-dot" style={{ background: dotColor[f.cls] }} />
                      <span className="feed-ip">{f.ip}</span>
                      <span className={`feed-label ${f.cls}`}>{f.label}</span>
                      <span className="feed-rps">{f.rps}</span>
                      <span className="feed-time">{f.time}</span>
                    </div>
                  ))}
                </div>

                <LiveDetectionCard />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DdosPage;
