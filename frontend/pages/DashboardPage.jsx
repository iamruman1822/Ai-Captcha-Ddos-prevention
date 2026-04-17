import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './DashboardPage.css';

/* ─── helpers ──────────────────────────────────────────────────────────── */
function timeAgo(iso) {
  if (!iso) return '';
  const d = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (d < 5) return 'just now';
  if (d < 60) return `${d}s ago`;
  return `${Math.floor(d / 60)}m ago`;
}

/* ─── component ────────────────────────────────────────────────────────── */
function DashboardPage({ onSignOut }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  /* stats cards */
  const [totalReqs, setTotalReqs] = useState(0);
  const [ddosBlocked, setDdosBlocked] = useState(0);
  const [botsCaught, setBotsCaught] = useState(0);
  const [threatLevel, setThreatLevel] = useState({ val: 'LOW', sub: 'no active threats', color: '#4a9eff' });
  const [alertBadge, setAlertBadge] = useState('READY');

  /* simulation */
  const [simRunning, setSimRunning] = useState(false);
  const [simTag, setSimTag] = useState('READY');
  const [simLog, setSimLog] = useState([]);
  const [simProgress, setSimProgress] = useState(0);
  const pollRef = useRef(null);

  /* chart — traffic utilisation over time */
  const [chartData, setChartData] = useState([
    { l: '60m', v: 12, t: 'normal' },
    { l: '55m', v: 15, t: 'normal' },
    { l: '50m', v: 11, t: 'normal' },
    { l: '45m', v: 14, t: 'normal' },
    { l: '40m', v: 10, t: 'normal' },
    { l: '35m', v: 13, t: 'normal' },
    { l: '30m', v: 12, t: 'normal' },
    { l: '25m', v: 11, t: 'normal' },
    { l: '20m', v: 14, t: 'normal' },
    { l: '15m', v: 10, t: 'normal' },
    { l: '10m', v: 13, t: 'normal' },
    { l: '5m',  v: 12, t: 'normal' },
    { l: 'now', v: 11, t: 'normal' },
  ]);

  /* live classification feed (merged captcha + ddos sessions) */
  const [feedItems, setFeedItems] = useState([]);

  const navigate = useNavigate();
  const location = useLocation();

  const ipData = useMemo(
    () => [
      { ip: '185.220.101.5', cc: 'RU', score: 2, status: 'blocked' },
      { ip: '45.155.205.22', cc: 'CN', score: 8, status: 'blocked' },
      { ip: '23.129.64.217', cc: 'US', score: 33, status: 'watch' },
      { ip: '103.21.244.8', cc: 'IN', score: 76, status: 'clean' },
      { ip: '91.108.4.118', cc: 'DE', score: 88, status: 'clean' },
      { ip: '176.10.99.200', cc: 'GB', score: 94, status: 'clean' },
    ],
    [],
  );

  const scoreColor = (score) => {
    if (score < 20) return '#ff5555';
    if (score < 50) return '#f59e0b';
    return '#00d4aa';
  };

  const dotColors = {
    human: '#00d4aa',
    bot: '#ff5555',
    suspicious: '#f59e0b',
    benign: '#00d4aa',
    attack: '#ff5555',
  };

  /* ── Poll both session endpoints for the live feed ──────────────── */
  const fetchFeed = useCallback(async () => {
    try {
      const [captchaRes, ddosRes] = await Promise.all([
        fetch('/api/captcha/sessions?limit=4'),
        fetch('/api/ddos/sessions?limit=4'),
      ]);
      const captcha = captchaRes.ok ? await captchaRes.json() : [];
      const ddos = ddosRes.ok ? await ddosRes.json() : [];

      const merged = [
        ...captcha.map((s) => ({
          ip: s.ip,
          label: s.prediction?.toUpperCase() || '?',
          cls: s.prediction === 'human' ? 'human' : s.prediction === 'bot' ? 'bot' : 'suspicious',
          conf: s.confidence || '—',
          time: timeAgo(s.timestamp),
          ts: new Date(s.timestamp).getTime(),
        })),
        ...ddos.map((s) => ({
          ip: s.ip,
          label: s.classification?.toUpperCase() || '?',
          cls: s.classification === 'attack' ? 'attack' : 'benign',
          conf: s.confidence || '—',
          time: timeAgo(s.timestamp),
          ts: new Date(s.timestamp).getTime(),
        })),
      ].sort((a, b) => b.ts - a.ts).slice(0, 7);

      if (merged.length > 0) setFeedItems(merged);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    fetchFeed();
    const t = setInterval(fetchFeed, 2500);
    return () => clearInterval(t);
  }, [fetchFeed]);

  /* ── Simulation polling ────────────────────────────────────────── */
  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/simulate/status');
      if (!res.ok) return;
      const d = await res.json();

      setSimProgress(d.progress || 0);
      setTotalReqs(d.total_requests || 0);
      setDdosBlocked(d.ddos_blocked || 0);
      setBotsCaught(d.bots_caught || 0);

      /* log lines */
      if (d.logs) setSimLog(d.logs);

      /* threat level */
      if (d.phase === 'attack') {
        setThreatLevel({ val: 'HIGH', sub: 'DDoS + bot attack in progress', color: '#ff5555' });
        setAlertBadge('⚠ ATTACK LIVE');
        setChartData((prev) => {
          const n = [...prev];
          n[11] = { l: '5m', v: 92, t: 'attack' };
          n[12] = { l: 'now', v: 98, t: 'attack' };
          return n;
        });
      } else if (d.phase === 'cooldown' || d.phase === 'done') {
        setThreatLevel({ val: 'LOW', sub: 'attack contained', color: '#00d4aa' });
        setAlertBadge('✓ CONTAINED');
        setChartData((prev) => {
          const n = [...prev];
          n[11] = { l: '5m', v: 30, t: 'normal' };
          n[12] = { l: 'now', v: 18, t: 'normal' };
          return n;
        });
      }

      /* stop polling if done */
      if (!d.running && d.phase === 'done') {
        setSimRunning(false);
        setSimTag('DONE');
        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
        // Reset after 5 seconds
        setTimeout(() => {
          setSimTag('READY');
          setThreatLevel({ val: 'LOW', sub: 'no active threats', color: '#4a9eff' });
          setAlertBadge('READY');
        }, 5000);
      }
    } catch {
      /* ignore */
    }
  }, []);

  /* ── Simulation trigger ────────────────────────────────────────── */
  const runSim = async () => {
    if (simRunning) return;

    setSimRunning(true);
    setSimTag('RUNNING');
    setSimLog([]);
    setSimProgress(0);
    setTotalReqs(0);
    setDdosBlocked(0);
    setBotsCaught(0);
    setThreatLevel({ val: 'MEDIUM', sub: 'simulation starting…', color: '#f59e0b' });
    setAlertBadge('▶ SIMULATING');

    try {
      const res = await fetch('/api/simulate/attack', { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        setSimLog([{ cls: 'err', text: `> Error: ${err.error || 'failed to start'}` }]);
        setSimRunning(false);
        setSimTag('READY');
        return;
      }

      // Start polling
      pollRef.current = setInterval(pollStatus, 500);
    } catch (e) {
      setSimLog([{ cls: 'err', text: `> Network error: ${e.message}` }]);
      setSimRunning(false);
      setSimTag('READY');
    }
  };

  /* cleanup on unmount */
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  return (
    <div className="dashboard-page">
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
              <span className="tb-page">dashboard</span>
              <span className="tb-sep">·</span>
              <span className="tb-site">myshop.com</span>
            </div>
            <div className="tb-right">
              <span className={`chip ${simRunning ? 'chip-alert' : 'chip-green'}`}>{alertBadge}</span>
              <span className="chip chip-green">PROTECTED</span>
              <span className="chip chip-neutral">sk_live_a8x2...</span>
              <div className="sb-avatar" onClick={onSignOut}>TK</div>
            </div>
          </div>

          <div className="content">
            {/* ── Stats Grid ──────────────────────────────────── */}
            <div className="stats-grid">
              <div className="stat-card green">
                <div className="stat-label">Total Requests</div>
                <div className="stat-value green">{totalReqs}</div>
                <div className="stat-sub">{simRunning ? 'simulation active' : 'run simulation'}</div>
              </div>
              <div className="stat-card red">
                <div className="stat-label">DDoS Blocked</div>
                <div className="stat-value red">{ddosBlocked}</div>
                <div className="stat-sub">XGBoost engine</div>
              </div>
              <div className="stat-card amber">
                <div className="stat-label">Bots Caught</div>
                <div className="stat-value amber">{botsCaught}</div>
                <div className="stat-sub">LSTM CAPTCHA engine</div>
              </div>
              <div className="stat-card blue">
                <div className="stat-label">Threat Level</div>
                <div className="stat-value blue" style={{ color: threatLevel.color }}>{threatLevel.val}</div>
                <div className="stat-sub">{threatLevel.sub}</div>
              </div>
            </div>

            {/* ── Two columns: chart + feed ────────────────────── */}
            <div className="two-col">
              <div className="panel">
                <div className="panel-header">
                  <span className="panel-title">Traffic · last 60 min</span>
                  <span className="panel-tag">LIVE</span>
                </div>
                <div className="chart-body">
                  {chartData.map((b) => (
                    <div className="bar-row" key={b.l}>
                      <span className="bar-label">{b.l}</span>
                      <div className="bar-track">
                        <div className={`bar-fill ${b.t}`} style={{ width: `${b.v}%` }} />
                      </div>
                      <span className="bar-value">{b.v}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <span className="panel-title">Classification Feed</span>
                  <span className="panel-tag">LIVE</span>
                </div>
                <div className="feed-body">
                  {feedItems.length === 0 && (
                    <div style={{ fontSize: 11, color: '#6b7280', textAlign: 'center', padding: '24px 0' }}>
                      No classifications yet — run a simulation
                    </div>
                  )}
                  {feedItems.map((f, idx) => (
                    <div className="feed-item" key={`${f.ip}-${idx}`}>
                      <div className="feed-dot" style={{ background: dotColors[f.cls] || '#8892a8' }} />
                      <span className="feed-ip">{f.ip}</span>
                      <span className={`feed-label ${f.cls}`}>{f.label}</span>
                      <span className="feed-conf">{f.conf}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* ── Bottom row: IP trust + Simulator ─────────────── */}
            <div className="bottom-row">
              <div className="panel">
                <div className="panel-header">
                  <span className="panel-title">IP trust scores</span>
                  <span style={{ fontSize: '9px', color: '#383848' }}>6 tracked</span>
                </div>
                <div className="ip-list">
                  {ipData.map((ip) => (
                    <div className="ip-row" key={ip.ip}>
                      <span className="ip-addr">{ip.ip}</span>
                      <span className="ip-country">{ip.cc}</span>
                      <div className="ip-bar-track">
                        <div className="ip-bar-fill" style={{ width: `${ip.score}%`, background: scoreColor(ip.score) }} />
                      </div>
                      <span className="ip-score" style={{ color: scoreColor(ip.score) }}>{ip.score}</span>
                      <span className={`ip-badge ${ip.status}`}>{ip.status.toUpperCase()}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="panel-header">
                  <span className="panel-title">Attack Simulator</span>
                  <span className={`panel-tag ${simTag === 'RUNNING' ? 'red' : simTag === 'DONE' ? 'green' : ''}`}>{simTag}</span>
                </div>
                <div className="sim-body">
                  <p className="sim-desc">
                    Fire a controlled DDoS + bot attack through the real ML models.
                    All results are classified by XGBoost (DDoS) and LSTM (CAPTCHA).
                  </p>

                  {/* Progress bar */}
                  {simRunning && (
                    <div className="sim-progress-track">
                      <div className="sim-progress-fill" style={{ width: `${simProgress}%` }} />
                    </div>
                  )}

                  <button className="sim-btn" disabled={simRunning} onClick={runSim}>
                    {simRunning ? `⏳ ${simProgress}% — SIMULATING…` : '▶  SIMULATE ATTACK'}
                  </button>

                  <div className="sim-log">
                    {simLog.map((entry, idx) => (
                      <div className={entry.cls} key={`${idx}-${entry.text?.slice(0, 10)}`}>{entry.text}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
