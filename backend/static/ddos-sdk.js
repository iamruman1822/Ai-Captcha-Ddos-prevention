/**
 * ZeroDay DDoS Shield SDK — Server-side DDoS detection integration
 *
 * Usage (paste before </body>):
 *
 *   <script
 *     src="http://localhost:5000/api/ddos/sdk.js"
 *     data-api-url="http://localhost:5000"
 *     data-api-key="YOUR_API_KEY"
 *     data-site-id="YOUR_SITE_ID">
 *   </script>
 *
 * This SDK monitors client-side request behaviour (navigation timing,
 * request rates) and sends lightweight telemetry to the DDoS detection API.
 *
 * For full accuracy, integrate the REST API server-side where real network
 * flow features (packet rates, SYN flags, IATs) are available.
 *
 * Result is dispatched as a CustomEvent "zerodayDdosResult" on `document`.
 * Also stored at window.__ZERODAY_DDOS__
 */
;(function () {
  'use strict';

  /* ── Configuration from script tag ─────────────────────────────────── */
  var scriptTag =
    document.currentScript ||
    document.querySelector('script[data-api-url]');

  var API_URL = (scriptTag && scriptTag.getAttribute('data-api-url')) || 'http://localhost:5000';
  var API_KEY = (scriptTag && scriptTag.getAttribute('data-api-key')) || '';
  var SITE_ID = (scriptTag && scriptTag.getAttribute('data-site-id')) || '';

  /* ── Settings ──────────────────────────────────────────────────────── */
  var CHECK_INTERVAL = 10000; // check every 10 seconds

  /* ── State ─────────────────────────────────────────────────────────── */
  var requestCount = 0;
  var startTime = Date.now();

  window.__ZERODAY_DDOS__ = null;

  /* ── Track page activity ───────────────────────────────────────────── */
  // Count XHR / fetch requests as a proxy for request rate
  var origOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function () {
    requestCount++;
    return origOpen.apply(this, arguments);
  };

  if (window.fetch) {
    var origFetch = window.fetch;
    window.fetch = function () {
      requestCount++;
      return origFetch.apply(this, arguments);
    };
  }

  /* ── Build telemetry & send ────────────────────────────────────────── */
  function check() {
    var elapsed = (Date.now() - startTime) / 1000;
    var rps = requestCount / Math.max(elapsed, 1);

    // Build a simplified flow feature set from browser telemetry
    var timing = performance.timing || {};
    var payload = {
      duration: elapsed,
      packets_rate: rps * 10,          // approximate
      bytes_rate: rps * 1500,          // approximate
      fwd_packets_rate: rps * 10,
      bwd_packets_rate: rps * 5,
      packet_IAT_std: rps > 5 ? 0.001 : 0.5,
      fwd_packets_IAT_std: rps > 5 ? 0.001 : 0.4,
      bwd_packets_IAT_std: rps > 5 ? 0.0 : 0.3,
      syn_flag_counts: rps > 10 ? Math.floor(rps * 3) : 1,
      ack_flag_counts: Math.max(1, Math.floor(rps)),
      down_up_rate: 1.0,
      bwd_packets_count: Math.floor(rps * 5),
      payload_bytes_mean: 700,
      payload_bytes_std: 300,
      payload_bytes_max: 1460,
      payload_bytes_min: 0,
      avg_segment_size: 700,
      fin_flag_counts: 1,
      rst_flag_counts: 0,
      urg_flag_counts: 0,
      cwr_flag_counts: 0,
      ece_flag_counts: 0,
      fwd_init_win_bytes: 65535,
      bwd_init_win_bytes: 65535,
      fwd_avg_segment_size: 700,
      bwd_avg_segment_size: 700,
      active_mean: elapsed / 2,
      active_std: 0.5,
      idle_mean: 5.0,
      idle_std: 1.0,
      bwd_total_header_bytes: rps * 200,
      subflow_fwd_bytes: rps * 1500,
      subflow_bwd_bytes: rps * 1000,
      syn_to_ack_ratio: rps > 10 ? rps * 0.3 : 0.02,
      fwd_bwd_rate_ratio: 1.5,
      iat_uniformity: rps > 5 ? 0.95 : 0.3,
      aggressive_flag_total: rps > 10 ? Math.floor(rps * 3) : 1,
      payload_range: 1460,
      window_asymmetry: 0,
      bytes_per_packet: 150,
      site_id: SITE_ID,
      api_key: API_KEY
    };

    var xhr = new XMLHttpRequest();
    xhr.open('POST', API_URL + '/api/ddos/detect', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onload = function () {
      if (xhr.status === 200) {
        try {
          var result = JSON.parse(xhr.responseText);
          window.__ZERODAY_DDOS__ = result;
          document.dispatchEvent(
            new CustomEvent('zerodayDdosResult', { detail: result })
          );
        } catch (_) {}
      }
    };

    xhr.send(JSON.stringify(payload));

    // Reset counters for next interval
    requestCount = 0;
    startTime = Date.now();
  }

  /* ── Timers ────────────────────────────────────────────────────────── */
  setTimeout(function () {
    check();
    setInterval(check, CHECK_INTERVAL);
  }, 3000);
})();
