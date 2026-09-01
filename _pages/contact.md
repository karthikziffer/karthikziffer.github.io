---
title: "Contact"
layout: page
permalink: "/contact.html"
---

<style>
.booking-card {
  border:1px solid #ececf1; border-radius:16px; padding:1.5rem 1.6rem 1rem; background:#fff;
  box-shadow:0 1px 3px rgba(0,0,0,.04); margin:1.4rem 0 0;
}
.booking-card h4 { margin:0 0 .3rem; font-weight:700; font-size:1.15rem; color:#222; display:flex; align-items:center; gap:.5rem; }
.booking-card h4 i { color:#0077b6; }
.booking-card p { margin:0 0 1rem; color:#555; font-size:.94rem; }
#cal-booking-inline { width:100%; height:650px; overflow:auto; border-radius:10px; }
@media (max-width:575px){ #cal-booking-inline { height:750px; } }
</style>

<div class="booking-card">
  <h4><i class="fas fa-calendar-alt"></i> Book a Meeting</h4>
  <p>Pick a time that works for you and it'll land straight on my calendar.</p>
  <div id="cal-booking-inline"></div>
  <script type="text/javascript">
    (function (C, A, L) { let p = function (a, ar) { a.q.push(ar); }; let d = C.document; C.Cal = C.Cal || function () { let cal = C.Cal; let ar = arguments; if (!cal.loaded) { cal.ns = {}; cal.q = cal.q || []; d.head.appendChild(d.createElement("script")).src = A; cal.loaded = true; } if (ar[0] === L) { const api = function () { p(api, arguments); }; const namespace = ar[1]; api.q = api.q || []; if(typeof namespace === "string"){cal.ns[namespace] = cal.ns[namespace] || api;p(cal.ns[namespace], ar);p(cal, ["initNamespace", namespace]);} else p(cal, ar); return;} p(cal, ar); }; })(window, "https://app.cal.com/embed/embed.js", "init");
    Cal("init", {origin:"https://cal.com"});

    Cal("inline", {
      elementOrSelector:"#cal-booking-inline",
      calLink: "karthik-bengl",
      layout: "month_view"
    });

    Cal("ui", {"styles":{"branding":{"brandColor":"#0077b6"}},"hideEventTypeDetails":false,"layout":"month_view"});
  </script>
</div>
