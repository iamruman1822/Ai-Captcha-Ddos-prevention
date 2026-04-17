import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './SettingsPage.css';

function SettingsPage({ onSignOut }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [emailAlerts, setEmailAlerts] = useState(false);
  const [strictMode, setStrictMode] = useState(false);
  const [frequency, setFrequency] = useState('realtime');

  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    setEmailAlerts(localStorage.getItem('pref-email') === '1');
    setStrictMode(localStorage.getItem('pref-strict') === '1');

    const savedFrequency = localStorage.getItem('pref-frequency');
    if (savedFrequency) {
      setFrequency(savedFrequency);
    }
  }, []);

  const toggleEmail = () => {
    setEmailAlerts((prev) => {
      const next = !prev;
      localStorage.setItem('pref-email', next ? '1' : '0');
      return next;
    });
  };

  const toggleStrict = () => {
    setStrictMode((prev) => {
      const next = !prev;
      localStorage.setItem('pref-strict', next ? '1' : '0');
      return next;
    });
  };

  const handleFrequencyChange = (event) => {
    const next = event.target.value;
    setFrequency(next);
    localStorage.setItem('pref-frequency', next);
  };

  const handleChangePassword = () => {
    window.alert('Change password flow - connect to backend endpoint.');
  };

  return (
    <div className="dashboard-page settings-page">
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
              <span className="tb-page">settings</span>
              <span className="tb-sep">·</span>
              <span className="tb-site">myshop.com</span>
            </div>
            <div className="tb-right">
              <span className="chip chip-green">PROTECTED</span>
              <span className="chip chip-neutral">sk_live_a8x2...</span>
              <div className="sb-avatar" onClick={onSignOut}>TK</div>
            </div>
          </div>

          <div className="page">
            <div className="settings-grid">
              <div className="stack-col">
                <div>
                  <p className="sec-label"><span className="sec-icon">USR</span>User details</p>

                  <div className="card">
                    <div className="row">
                      <span className="row-label">Name</span>
                      <span className="row-value">Tanya</span>
                    </div>
                    <div className="row">
                      <span className="row-label">Email</span>
                      <span className="row-value">tanya@example.com</span>
                    </div>
                    <div className="row">
                      <span className="row-label">Role</span>
                      <span className="row-value">Admin</span>
                    </div>
                    <div className="row">
                      <span className="row-label">Plan</span>
                      <span className="row-value">Team · Active</span>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="sec-label"><span className="sec-icon">ACC</span>Account</p>

                  <div className="card">
                    <div className="btn-row">
                      <button className="btn btn-neutral" onClick={handleChangePassword}>
                        CHANGE PASSWORD
                      </button>
                      <button className="btn btn-danger" onClick={onSignOut}>
                        LOGOUT
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="stack-col">
                <div>
                  <p className="sec-label"><span className="sec-icon">SEC</span>Security preferences</p>

                  <div className="card">
                    <div className="pref-row">
                      <div className="pref-copy">
                        <span className="pref-title">Email alerts</span>
                        <span className="pref-sub">Get incident notifications by email.</span>
                      </div>
                      <button className={`mini-switch ${emailAlerts ? 'on' : ''}`} onClick={toggleEmail} type="button" aria-label="Toggle email alerts">
                        <span className="knob" />
                      </button>
                    </div>

                    <div className="pref-row">
                      <div className="pref-copy">
                        <span className="pref-title">Strict mode</span>
                        <span className="pref-sub">Apply tougher filtering on suspicious traffic.</span>
                      </div>
                      <button className={`mini-switch ${strictMode ? 'on' : ''}`} onClick={toggleStrict} type="button" aria-label="Toggle strict mode">
                        <span className="knob" />
                      </button>
                    </div>

                    <div className="pref-row pref-row-inline">
                      <label className="pref-title" htmlFor="pref-frequency">Alert digest</label>
                      <select id="pref-frequency" className="input-select" value={frequency} onChange={handleFrequencyChange}>
                        <option value="realtime">Realtime</option>
                        <option value="hourly">Hourly</option>
                        <option value="daily">Daily</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="sec-label"><span className="sec-icon">LOG</span>Recent activity</p>

                  <div className="card">
                    <div className="activity-list">
                      <div className="activity-item">
                        <span className="activity-dot ok" />
                        <span className="activity-text">Changed alert digest to realtime</span>
                        <span className="activity-time">2m ago</span>
                      </div>
                      <div className="activity-item">
                        <span className="activity-dot warn" />
                        <span className="activity-text">Switched DDoS mode to strict</span>
                        <span className="activity-time">19m ago</span>
                      </div>
                      <div className="activity-item">
                        <span className="activity-dot ok" />
                        <span className="activity-text">Updated account password</span>
                        <span className="activity-time">1h ago</span>
                      </div>
                    </div>
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

export default SettingsPage;
