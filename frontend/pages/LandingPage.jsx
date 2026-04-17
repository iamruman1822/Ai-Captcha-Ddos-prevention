import { useRef, useState, useEffect } from 'react';
import { supabase, isSupabaseConfigured } from '../lib/supabaseClient';
import './LandingPage.css';

function LandingPage({ session, onLoginSuccess }) {
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [authFeedback, setAuthFeedback] = useState('');
  const [authError, setAuthError] = useState('');
  const authDropdownRef = useRef(null);

  useEffect(() => {
    if (!isAuthOpen) {
      return undefined;
    }

    const handleOutsideClick = (event) => {
      if (authDropdownRef.current && !authDropdownRef.current.contains(event.target)) {
        setIsAuthOpen(false);
      }
    };

    document.addEventListener('mousedown', handleOutsideClick);

    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, [isAuthOpen]);

  const openAuthDropdown = (mode) => {
    setAuthMode(mode);
    setAuthFeedback('');
    setAuthError('');
    setIsAuthOpen(true);
  };

  const handleAuthSubmit = async (event) => {
    event.preventDefault();

    if (!isSupabaseConfigured || !supabase) {
      setAuthError('Supabase is not configured. Add your keys in frontend/.env.local.');
      return;
    }

    setAuthError('');
    setAuthFeedback('');
    setIsSubmitting(true);

    if (authMode === 'signup') {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: fullName.trim(),
          },
        },
      });

      setIsSubmitting(false);

      if (error) {
        setAuthError(error.message);
        return;
      }

      setAuthFeedback('Signup successful. Check your email to verify your account.');
      return;
    }

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    setIsSubmitting(false);

    if (error) {
      setAuthError(error.message);
      return;
    }

    setAuthFeedback('Login successful. Redirecting to dashboard...');
    setPassword('');
    setIsAuthOpen(false);
    onLoginSuccess();
  };

  const handleSignOut = async () => {
    if (!supabase) {
      return;
    }

    await supabase.auth.signOut();
    setAuthFeedback('You have been signed out.');
    setAuthError('');
  };

  const userEmail = session?.user?.email;

  return (
    <>
      <nav className="navbar">
        <div className="brand-wrap">
          <span className="logo-text">
            ZERO<span className="accent">DAY</span>
          </span>
        </div>

        <div className="nav-links">
          <button className="nav-link" type="button">PRODUCT</button>
          <button className="nav-link" type="button">DOCS</button>
          <button className="nav-link" type="button">PRICING</button>
        </div>

        <div className="nav-right">
          {!userEmail && (
            <button className="nav-link" type="button" onClick={() => openAuthDropdown('login')}>
              LOGIN
            </button>
          )}

          {userEmail ? (
            <button className="btn-nav" type="button" onClick={handleSignOut}>
              LOGOUT
            </button>
          ) : (
            <button className="btn-nav" type="button" onClick={() => openAuthDropdown('signup')}>
              GET STARTED -&gt;
            </button>
          )}

          {isAuthOpen && (
            <div className="auth-dropdown" ref={authDropdownRef} role="dialog" aria-modal="false">
              <div className="auth-header">
                <p className="auth-title">{authMode === 'login' ? 'Account Login' : 'Create Account'}</p>
                <button
                  className="auth-close"
                  type="button"
                  onClick={() => setIsAuthOpen(false)}
                  aria-label="Close authentication panel"
                >
                  x
                </button>
              </div>

              <div className="auth-switcher">
                <button
                  className={`auth-tab ${authMode === 'login' ? 'active' : ''}`}
                  type="button"
                  onClick={() => setAuthMode('login')}
                >
                  LOGIN
                </button>
                <button
                  className={`auth-tab ${authMode === 'signup' ? 'active' : ''}`}
                  type="button"
                  onClick={() => setAuthMode('signup')}
                >
                  SIGNUP
                </button>
              </div>

              <form className="auth-form" onSubmit={handleAuthSubmit}>
                {authMode === 'signup' && (
                  <label className="auth-label" htmlFor="fullName">
                    Full Name
                    <input
                      id="fullName"
                      className="auth-input"
                      type="text"
                      value={fullName}
                      onChange={(event) => setFullName(event.target.value)}
                      placeholder="Jane Doe"
                      required
                    />
                  </label>
                )}

                <label className="auth-label" htmlFor="email">
                  Email
                  <input
                    id="email"
                    className="auth-input"
                    type="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    placeholder="you@company.com"
                    required
                  />
                </label>

                <label className="auth-label" htmlFor="password">
                  Password
                  <input
                    id="password"
                    className="auth-input"
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Minimum 6 characters"
                    minLength={6}
                    required
                  />
                </label>

                {authError && <p className="auth-error">{authError}</p>}
                {authFeedback && <p className="auth-success">{authFeedback}</p>}

                {!isSupabaseConfigured && (
                  <p className="auth-hint">
                    Add REACT_APP_SUPABASE_URL and REACT_APP_SUPABASE_ANON_KEY in frontend/.env.local.
                  </p>
                )}

                <button className="auth-submit" type="submit" disabled={isSubmitting}>
                  {isSubmitting ? 'PLEASE WAIT...' : authMode === 'login' ? 'LOGIN' : 'CREATE ACCOUNT'}
                </button>
              </form>
            </div>
          )}
        </div>
      </nav>

      <section className="hero">
        <div className="grid-bg" />

        <div className="hero-content">
          <div className="hero-tag">
            <span className="dot pulse-dot" />
            AI-POWERED · REAL-TIME · ADAPTIVE
          </div>

          <h1 className="hero-title">
            <span className="line-white">Stop bots.</span>
            <br />
            <span className="line-teal">Kill DDoS.</span>
            <br />
            <span className="line-dim">Let humans through.</span>
          </h1>

          <p className="hero-sub">
            ZeroDay is an intelligent security layer that uses
            <span className="highlight"> machine learning</span>
            {' '}to silently classify every visitor - blocking bots
            and attacks before they reach your server.
          </p>

          <p className="backend-status">Authentication ready with Supabase.</p>
          {userEmail && <p className="backend-status">Logged in as: {userEmail}</p>}

          <div className="hero-actions">
            <button className="btn-primary" type="button">START FOR FREE -&gt;</button>

            <button className="btn-secondary" type="button">
              <span className="play-box" aria-hidden="true">
                <svg width="6" height="8" viewBox="0 0 6 8" fill="none">
                  <path className="play-icon" d="M0 0L6 4L0 8V0Z" />
                </svg>
              </span>
              VIEW DEMO
            </button>
          </div>
        </div>
      </section>

      <section className="how-section">
        <p className="section-tag">HOW IT WORKS</p>
        <h2 className="section-title">One script tag. Full protection.</h2>
        <p className="section-sub">
          Add one line to your site. ZeroDay handles everything -
          silently, in real time, with zero friction for real users.
        </p>

        <div className="steps-grid">
          <div className="step-card">
            <p className="step-number">01</p>

            <div className="step-icon-box">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <rect x="1" y="1" width="14" height="14" stroke="#00d4aa" strokeWidth="1.2" />
                <path d="M4 8h8M8 4v8" stroke="#00d4aa" strokeWidth="1.2" />
              </svg>
            </div>

            <p className="step-title">Register your site</p>
            <p className="step-desc">
              Sign up, add your domain, and get a unique API key
              and SDK snippet generated instantly.
            </p>
            <span className="step-badge">30 SECONDS</span>
          </div>

          <div className="step-card">
            <p className="step-number">02</p>

            <div className="step-icon-box">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M2 4h12M2 8h8M2 12h5" stroke="#00d4aa" strokeWidth="1.2" />
              </svg>
            </div>

            <p className="step-title">Paste the script tag</p>
            <p className="step-desc">
              Drop one &lt;script&gt; tag into your HTML.
              Works on any site - WordPress, React,
              plain HTML, anything.
            </p>
            <span className="step-badge">1 LINE OF CODE</span>
          </div>

          <div className="step-card">
            <p className="step-number">03</p>

            <div className="step-icon-box">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M8 1L2 5v5c0 3 2.5 5.5 6 5.5s6-2.5 6-5.5V5L8 1z" stroke="#00d4aa" strokeWidth="1.2" />
              </svg>
            </div>

            <p className="step-title">Protection is live</p>
            <p className="step-desc">
              ML models classify every visitor in real time.
              Bots get blocked. Humans pass through seamlessly.
            </p>
            <span className="step-badge">ZERO FRICTION</span>
          </div>
        </div>
      </section>

      <footer className="footer">
        <span className="footer-logo-text">
          ZERO<span className="accent">DAY</span>
        </span>

        <div className="footer-links">
          <button className="footer-link" type="button">DOCS</button>
          <button className="footer-link" type="button">PRIVACY</button>
          <button className="footer-link" type="button">GITHUB</button>
        </div>

        <span className="footer-copy">© 2026 ZERODAY</span>
      </footer>
    </>
  );
}

export default LandingPage;
