/**
 * ZeroDay Captcha SDK — Behaviour-based bot detection
 *
 * Usage (paste before </body>):
 *
 *   <script
 *     src="http://localhost:5000/api/captcha/sdk.js"
 *     data-api-url="http://localhost:5000"
 *     data-api-key="YOUR_API_KEY"
 *     data-site-id="YOUR_SITE_ID">
 *   </script>
 *
 * The SDK silently tracks mouse movements, sends them to the API after a
 * short collection window, and dispatches a CustomEvent "zerodayCaptchaResult"
 * on `document` with the classification payload.
 *
 * You can also check the result programmatically:
 *   window.__ZERODAY_CAPTCHA__   → latest result object (or null)
 */
;(function () {
  'use strict';

  /* ── Configuration from script tag ─────────────────────────────────── */
  var scriptTag =
    document.currentScript ||
    document.querySelector('script[data-api-url]');

  var API_URL  = (scriptTag && scriptTag.getAttribute('data-api-url'))  || 'http://localhost:5000';
  var API_KEY  = (scriptTag && scriptTag.getAttribute('data-api-key'))  || '';
  var SITE_ID  = (scriptTag && scriptTag.getAttribute('data-site-id'))  || '';

  /* ── Settings ──────────────────────────────────────────────────────── */
  var COLLECT_MS       = 3000;   // collect movements for 3 seconds
  var MIN_POINTS       = 15;     // minimum coordinates before sending
  var REPEAT_INTERVAL  = 10000;  // re-check every 10 s (0 = one-shot)

  /* ── State ─────────────────────────────────────────────────────────── */
  var coords  = [];
  var sending = false;

  window.__ZERODAY_CAPTCHA__ = null;

  /* ── Collect mouse movements ───────────────────────────────────────── */
  function onMouseMove(e) {
    coords.push([e.clientX, e.clientY]);
  }
  document.addEventListener('mousemove', onMouseMove);

  /* ── Send to API ───────────────────────────────────────────────────── */
  function classify() {
    if (sending || coords.length < MIN_POINTS) return;
    sending = true;

    var payload = {
      mouse_movements: coords.slice(),   // copy
      site_id: SITE_ID,
      api_key: API_KEY
    };

    // Reset for next round
    coords = [];

    var xhr = new XMLHttpRequest();
    xhr.open('POST', API_URL + '/api/captcha/verify', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onload = function () {
      sending = false;
      if (xhr.status === 200) {
        try {
          var result = JSON.parse(xhr.responseText);
          window.__ZERODAY_CAPTCHA__ = result;
          document.dispatchEvent(
            new CustomEvent('zerodayCaptchaResult', { detail: result })
          );
        } catch (_) { /* ignore parse errors */ }
      }
    };

    xhr.onerror = function () { sending = false; };
    xhr.send(JSON.stringify(payload));
  }

  /* ── Timers ────────────────────────────────────────────────────────── */
  // First classification after COLLECT_MS
  setTimeout(function () {
    classify();
    // Optionally repeat
    if (REPEAT_INTERVAL > 0) {
      setInterval(classify, REPEAT_INTERVAL);
    }
  }, COLLECT_MS);
})();
