---
title: "About"
layout: page
permalink: "/about.html"
---

<style>
.about-page { color:#2b2b2b; }
.about-page h4.spanborder { margin-top:2.25rem; }

/* Hero */
.about-hero {
  border-radius:16px;
  background:linear-gradient(135deg,#03045e 0%,#0077b6 55%,#00b4d8 100%);
  color:#fff; padding:2rem 2.1rem; margin:0 0 1.9rem;
  box-shadow:0 10px 30px rgba(0,119,182,.22);
}
.about-hero-inner { display:flex; align-items:center; gap:1.6rem; flex-wrap:wrap; }
.about-hero-text { flex:1; min-width:240px; }
.about-hero-text h1 { margin:0; font-weight:700; font-size:2rem; line-height:1.1; }
.about-role { font-size:1.18rem; margin:.25rem 0 .35rem; opacity:.96; }
.about-loc { margin:0 0 .85rem; font-size:.92rem; opacity:.85; }
.about-loc i { margin-right:.35rem; }
.about-links a {
  display:inline-flex; align-items:center; gap:.4rem; color:#fff;
  background:rgba(255,255,255,.16); padding:.34rem .8rem; border-radius:999px;
  margin:.22rem .45rem .22rem 0; font-size:.85rem; text-decoration:none;
  transition:background .18s ease, transform .18s ease;
}
.about-links a:hover { background:rgba(255,255,255,.34); transform:translateY(-1px); }

/* Summary + chips */
.about-summary { font-size:1.06rem; line-height:1.75; color:#333; margin-bottom:.6rem; text-align:justify; text-justify:inter-word; }
.chip-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:#9aa0ad; font-weight:700; margin:1rem 0 .5rem; }
.about-chips { display:flex; flex-wrap:wrap; gap:.5rem; }
.about-chips span {
  background:#caf0f8; color:#023e8a; border:1px solid #90e0ef;
  padding:.32rem .8rem; border-radius:999px; font-size:.84rem;
}

/* Info cards */
.about-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(225px,1fr)); gap:1rem; }
.about-card {
  border:1px solid #ececf1; border-radius:14px; padding:1.15rem 1.3rem; background:#fff;
  box-shadow:0 1px 3px rgba(0,0,0,.04);
}
.about-card h6 { font-weight:700; text-transform:uppercase; letter-spacing:.05em; font-size:.76rem; color:#8a90a0; margin-bottom:.7rem; }
.about-card ul { list-style:none; padding:0; margin:0; }
.about-card li { padding:.22rem 0; font-size:.94rem; line-height:1.4; }
.about-card li .lvl { color:#aab; font-size:.8rem; }

/* Skills — brand-icon pills */
.skill-pills { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:.5rem; }
.skill-pill { display:inline-flex; align-items:center; gap:.45rem; background:#fff; border:1px solid #e7eaf1; border-radius:999px; padding:.4rem .85rem; font-size:.9rem; color:#333; box-shadow:0 1px 2px rgba(0,0,0,.04); transition:transform .15s ease, box-shadow .15s ease; }
.skill-pill:hover { transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,.08); }
.skill-pill img { width:18px; height:18px; object-fit:contain; }
.skill-pill .skill-fallback { color:#0077b6; font-size:.95rem; }

/* Languages */
.lang-pills { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:.5rem; }
.lang-pill { background:#fff; border:1px solid #e7eaf1; border-radius:12px; padding:.55rem .95rem; box-shadow:0 1px 2px rgba(0,0,0,.04); min-width:118px; }
.lang-pill strong { display:block; font-size:.94rem; color:#222; }
.lang-pill span { font-size:.77rem; color:#9aa0ad; }

/* Certifications */
.cert-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(245px,1fr)); gap:1rem; margin-top:.5rem; }
.cert-card { display:flex; gap:.8rem; align-items:center; border:1px solid #ececf1; border-radius:14px; padding:.9rem 1.05rem; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); transition:transform .15s ease, box-shadow .15s ease; }
.cert-card:hover { transform:translateY(-2px); box-shadow:0 6px 16px rgba(0,0,0,.08); }
.cert-icon { flex:0 0 38px; width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:1.05rem; background:linear-gradient(135deg,#0096c7,#00b4d8); }
.cert-name { font-size:.92rem; font-weight:600; line-height:1.3; color:#333; }

/* Timeline */
.timeline { position:relative; margin:.4rem 0 0; }
.tl-item { display:flex; gap:1.05rem; padding:0 0 1.7rem; position:relative; }
.tl-item:not(:last-child)::before {
  content:""; position:absolute; left:23px; top:54px; bottom:2px; width:2px; background:#e9ecf4;
}
.tl-logo { flex:0 0 48px; }
.tl-logo img {
  width:48px; height:48px; border-radius:12px; object-fit:contain;
  background:#fff; border:1px solid #eceff5; padding:6px;
}
.tl-badge {
  width:48px; height:48px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-weight:700; color:#fff; font-size:1.35rem;
  background:linear-gradient(135deg,#0096c7,#0077b6);
}
.tl-body { flex:1; min-width:0; }
.tl-head h5 { margin:0; font-weight:700; font-size:1.08rem; }
.tl-company { font-weight:600; color:#0077b6; }
.tl-company a { color:#0077b6; text-decoration:none; }
.tl-company a:hover { text-decoration:underline; }
.tl-meta { display:block; color:#9aa0ad; font-size:.84rem; margin:.12rem 0 .5rem; }
.tl-points { margin:.3rem 0 0; padding-left:1.15rem; }
.tl-points li { margin:.28rem 0; line-height:1.55; color:#454545; font-size:.94rem; }
.thesis-card { margin-top:.7rem; border:1px solid #e7eaf1; border-left:4px solid #0077b6; border-radius:12px; padding:.95rem 1.1rem; background:linear-gradient(180deg,#fafbff,#fff); }
.thesis-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:#0077b6; font-weight:700; display:flex; align-items:center; gap:.4rem; margin-bottom:.35rem; }
.thesis-title { font-weight:700; font-size:1rem; line-height:1.35; margin:0 0 .25rem; }
.thesis-title a { color:#222; text-decoration:none; }
.thesis-title a:hover { color:#0077b6; }
.thesis-obj { font-size:.88rem; color:#555; margin:0 0 .35rem; }
.thesis-obj strong { color:#333; }
.thesis-sup { font-size:.86rem; color:#555; margin:0 0 .55rem; }
.thesis-sup strong { color:#333; }
.thesis-sup a { color:#0077b6; text-decoration:none; font-weight:600; }
.thesis-sup a:hover { text-decoration:underline; }
ul.thesis-points { margin:.2rem 0 .65rem; padding-left:1.1rem; }
ul.thesis-points li { font-size:.9rem; line-height:1.5; color:#454545; margin:.24rem 0; }
.thesis-tags { display:flex; flex-wrap:wrap; gap:.4rem; }
.thesis-tags span { background:#caf0f8; color:#023e8a; border:1px solid #90e0ef; border-radius:999px; padding:.22rem .62rem; font-size:.76rem; }

/* Awards */
.awards-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(255px,1fr)); gap:1rem; margin-top:.4rem; }
.award-card { display:flex; gap:.85rem; border:1px solid #ececf1; border-radius:14px; padding:1.05rem 1.15rem; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); }
.award-icon { flex:0 0 42px; width:42px; height:42px; border-radius:11px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:1.15rem; background:linear-gradient(135deg,#f7971e,#ffc400); }
.award-body h5 { margin:0 0 .15rem; font-weight:700; font-size:1.02rem; line-height:1.25; }
.award-body .award-org { color:#9aa0ad; font-size:.82rem; display:block; margin-bottom:.35rem; }
.award-body p { margin:0 0 .35rem; font-size:.91rem; color:#555; line-height:1.5; }
.award-body a { font-size:.84rem; color:#0077b6; text-decoration:none; }
.award-body a:hover { text-decoration:underline; }

/* Industrial projects */
.proj-card { border:1px solid #ececf1; border-radius:14px; padding:1.2rem 1.35rem; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); margin-top:.4rem; }
.proj-card h5 { margin:0; font-weight:700; font-size:1.12rem; }
.proj-card h5 a { color:#222; text-decoration:none; }
.proj-card h5 a:hover { color:#0077b6; }
.proj-card .proj-org { color:#0077b6; font-weight:600; font-size:.9rem; }
.proj-card p { margin:.5rem 0 .95rem; color:#454545; line-height:1.6; font-size:.95rem; }
.proj-partners-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:#9aa0ad; font-weight:700; margin-bottom:.6rem; }
.proj-partners { display:flex; flex-wrap:wrap; gap:.7rem; align-items:center; }
.proj-partner { display:flex; align-items:center; gap:.5rem; background:#f7f8fb; border:1px solid #eceff5; border-radius:10px; padding:.4rem .75rem; }
.proj-partner img { width:24px; height:24px; object-fit:contain; }
.proj-partner span { font-size:.83rem; color:#444; }

/* In the Press */
.press-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:1rem; margin-top:.4rem; }
.press-card { display:flex; gap:.8rem; border:1px solid #ececf1; border-radius:14px; padding:1rem 1.1rem; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); transition:box-shadow .18s ease, transform .18s ease; }
.press-card:hover { box-shadow:0 6px 18px rgba(0,0,0,.08); transform:translateY(-2px); }
.press-logo { flex:0 0 38px; width:38px; height:38px; border-radius:9px; border:1px solid #eceff5; background:#fff; object-fit:contain; padding:5px; }
.press-body { min-width:0; }
.press-src { font-size:.77rem; color:#9aa0ad; font-weight:600; display:block; margin-bottom:.22rem; }
.press-body a { color:#222; font-weight:700; font-size:.95rem; line-height:1.35; text-decoration:none; }
.press-body a:hover { color:#0077b6; }

/* Open Source */
.os-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:1rem; margin-top:.4rem; }
.os-card { border:1px solid #ececf1; border-radius:14px; padding:1.1rem 1.25rem; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); transition:transform .15s ease, box-shadow .15s ease; }
.os-card:hover { transform:translateY(-2px); box-shadow:0 6px 16px rgba(0,0,0,.08); }
.os-head { display:flex; align-items:center; gap:.55rem; }
.os-head i { color:#222; font-size:1.2rem; }
.os-head a { color:#222; font-weight:700; font-size:1.05rem; text-decoration:none; }
.os-head a:hover { color:#0077b6; }
.os-card p { margin:.55rem 0 .6rem; color:#454545; line-height:1.55; font-size:.94rem; }
.os-meta { font-size:.83rem; color:#9aa0ad; }
.os-meta .os-lang-dot { color:#0077b6; margin-right:.15rem; }

@media (max-width:575px){
  .about-hero { padding:1.5rem 1.25rem; }
  .about-hero-text h1 { font-size:1.6rem; }
}
</style>

<div class="about-page">

  <div class="about-hero">
    <div class="about-hero-inner">
      <div class="about-hero-text">
        <h1>Karthik Rajendran</h1>
        <p class="about-role">AI / Machine Learning Engineer</p>
        <p class="about-loc"><i class="fas fa-map-marker-alt"></i>Erlangen, Bavaria, Germany &nbsp;·&nbsp; M.Sc, Universität Stuttgart</p>
        <div class="about-links">
          <a href="https://www.linkedin.com/in/karthikrks/" target="_blank" rel="noopener"><i class="fab fa-linkedin"></i> LinkedIn</a>
          <a href="https://github.com/karthikziffer" target="_blank" rel="noopener"><i class="fab fa-github"></i> GitHub</a>
          <a href="https://twitter.com/karthikziffer" target="_blank" rel="noopener"><i class="fab fa-twitter"></i> Twitter</a>
          <a href="mailto:karthik.bengaluru95@gmail.com"><i class="fas fa-envelope"></i> Email</a>
        </div>
      </div>
    </div>
  </div>

  <p class="about-summary">
    I design and build AI/ML models end to end, from early <strong>ideation to production</strong> systems serving thousands of requests. My work starts with understanding the business domain and designing for its specifics, then accounting for where probabilistic models fall short by adding <strong>deterministic guardrails</strong> and monitoring to keep performance reliable in production.
  </p>
  <p class="about-summary">
    Alongside the engineering, I drive projects forward, work closely with collaborators, and guide interns and master's thesis students.
  </p>
  <p class="about-summary">
    Outside of work, you'll usually find me playing badminton or tennis, or out hiking.
  </p>

  <div class="chip-label">Industries</div>
  <div class="about-chips">
    <span>Textile Recycling</span><span>Automotive</span><span>Material Science</span><span>Sensor Systems</span><span>Retail</span><span>Media</span><span>Education</span>
  </div>

  <div class="chip-label">Applications</div>
  <div class="about-chips">
    <span>Agentic Workflows</span><span>Agentic Chatbots</span><span>Optical Character Recognition</span><span>Demand &amp; Price Forecasting</span><span>Automated Textile Sorting</span><span>Anomaly Detection</span><span>Recommendation Systems</span><span>Risk Assessment</span><span>Document Information Retrieval</span>
  </div>

  <h4 class="font-weight-bold spanborder"><span>Skills</span></h4>
  <div class="skill-pills">
    {% for s in site.data.skills %}
    <span class="skill-pill">
      {% if s.icon != "" %}<img src="{{ site.baseurl }}/assets/images/{{ s.icon }}" alt="" loading="lazy" onerror="this.onerror=null;this.outerHTML='<i class=&quot;fas fa-cube skill-fallback&quot;></i>'">{% else %}<i class="fas fa-layer-group skill-fallback"></i>{% endif %}
      {{ s.name }}
    </span>
    {% endfor %}
  </div>

  <h4 class="font-weight-bold spanborder"><span>Languages</span></h4>
  <div class="lang-pills">
    <div class="lang-pill"><strong>German</strong><span>Limited Working</span></div>
    <div class="lang-pill"><strong>English</strong><span>Full Professional</span></div>
    <div class="lang-pill"><strong>Tamil</strong><span>Native</span></div>
    <div class="lang-pill"><strong>Kannada</strong><span>Native</span></div>
    <div class="lang-pill"><strong>Hindi</strong><span>Professional</span></div>
  </div>

  <h4 class="font-weight-bold spanborder"><span>Certifications</span></h4>
  <div class="cert-grid">
    <div class="cert-card"><span class="cert-icon"><i class="fas fa-certificate"></i></span><span class="cert-name">Neural Networks &amp; Deep Learning</span></div>
    <div class="cert-card"><span class="cert-icon"><i class="fas fa-certificate"></i></span><span class="cert-name">LLM apps with LangChain &amp; OpenAI</span></div>
    <div class="cert-card"><span class="cert-icon"><i class="fas fa-certificate"></i></span><span class="cert-name">Azure Machine Learning Fundamentals</span></div>
    <div class="cert-card"><span class="cert-icon"><i class="fas fa-certificate"></i></span><span class="cert-name">Azure Data Factory</span></div>
    <div class="cert-card"><span class="cert-icon"><i class="fas fa-certificate"></i></span><span class="cert-name">GDPR for Businesses &amp; Individuals</span></div>
  </div>

  <h4 class="font-weight-bold spanborder"><span>Experience</span></h4>
  <div class="timeline">
    {% for job in site.data.experience %}
    <div class="tl-item">
      <div class="tl-logo">
        {% if job.logo %}
        <img src="{{ site.baseurl }}/assets/images/companies/{{ job.slug }}.png" alt="{{ job.company }} logo" loading="lazy" onerror="this.onerror=null;this.parentNode.innerHTML='<span class=tl-badge>{{ job.company | slice: 0,1 }}</span>'">
        {% else %}
        <span class="tl-badge">{{ job.company | slice: 0,1 }}</span>
        {% endif %}
      </div>
      <div class="tl-body">
        <div class="tl-head">
          <h5>{{ job.role }}</h5>
          <span class="tl-company">{% if job.url %}<a href="{{ job.url }}" target="_blank" rel="noopener">{{ job.company }}</a>{% else %}{{ job.company }}{% endif %}</span>
          <span class="tl-meta">{{ job.start }} – {{ job.end }}{% if job.location != "" %} · {{ job.location }}{% endif %}</span>
        </div>
        {% if job.highlights %}
        <ul class="tl-points">
          {% for h in job.highlights %}<li>{{ h }}</li>{% endfor %}
        </ul>
        {% endif %}
      </div>
    </div>
    {% endfor %}
  </div>

  <h4 class="font-weight-bold spanborder"><span>Industrial Projects</span></h4>
  {% for proj in site.data.projects %}
  <div class="proj-card">
    <h5>{% if proj.url %}<a href="{{ proj.url }}" target="_blank" rel="noopener">{{ proj.name }}</a>{% else %}{{ proj.name }}{% endif %}</h5>
    <span class="proj-org">{{ proj.org }}</span>
    <p>{{ proj.desc }}</p>
    {% if proj.partners %}
    <div class="proj-partners-label">Project Partners</div>
    <div class="proj-partners">
      {% for p in proj.partners %}
      <div class="proj-partner"><img src="{{ site.baseurl }}/assets/images/{{ p.logo }}" alt="{{ p.name }}" loading="lazy" onerror="this.style.display='none'"><span>{{ p.name }}</span></div>
      {% endfor %}
    </div>
    {% endif %}
  </div>
  {% endfor %}

  <h4 class="font-weight-bold spanborder"><span>Open Source</span></h4>
  <div class="os-grid">
    {% for repo in site.data.opensource %}
    <div class="os-card">
      <div class="os-head"><i class="fab fa-github"></i><a href="{{ repo.url }}" target="_blank" rel="noopener">{{ repo.name }}</a></div>
      <p>{{ repo.desc }}</p>
      <div class="os-meta">{% if repo.language %}<span class="os-lang-dot">●</span>{{ repo.language }}{% endif %}<span class="os-stars" data-repo="{{ repo.repo }}"></span></div>
    </div>
    {% endfor %}
  </div>
  <script>
  (function () {
    document.querySelectorAll('.os-stars').forEach(function (el) {
      var r = el.getAttribute('data-repo');
      if (!r) return;
      fetch('https://api.github.com/repos/' + r)
        .then(function (x) { return x.json(); })
        .then(function (d) {
          if (d && typeof d.stargazers_count === 'number') {
            el.innerHTML = ' &nbsp;·&nbsp; ★ ' + d.stargazers_count;
          }
        })
        .catch(function () {});
    });
  })();
  </script>

  <h4 class="font-weight-bold spanborder"><span>In the Press</span></h4>
  <div class="press-grid">
    {% for item in site.data.press %}
    <div class="press-card">
      {% if item.logo != "" %}<img class="press-logo" src="{{ site.baseurl }}/assets/images/{{ item.logo }}" alt="{{ item.source }}" loading="lazy" onerror="this.onerror=null;this.outerHTML='<span class=press-logo style=&quot;display:flex;align-items:center;justify-content:center;color:#9aa0ad&quot;><i class=&quot;fas fa-newspaper&quot;></i></span>'">{% else %}<span class="press-logo" style="display:flex;align-items:center;justify-content:center;color:#9aa0ad"><i class="fas fa-newspaper"></i></span>{% endif %}
      <div class="press-body">
        <span class="press-src">{{ item.source }}{% if item.date != "" %} · {{ item.date }}{% endif %}</span>
        <a href="{{ item.url }}" target="_blank" rel="noopener">{{ item.title }}</a>
      </div>
    </div>
    {% endfor %}
  </div>

  <h4 class="font-weight-bold spanborder"><span>Awards</span></h4>
  <div class="awards-grid">
    {% for award in site.data.awards %}
    <div class="award-card">
      <div class="award-icon"><i class="fas fa-trophy"></i></div>
      <div class="award-body">
        <h5>{{ award.title }}</h5>
        <span class="award-org">{{ award.org }}</span>
        <p>{{ award.desc }}</p>
        {% if award.url %}<a href="{{ award.url }}" target="_blank" rel="noopener">View <i class="fas fa-external-link-alt" style="font-size:.7rem"></i></a>{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>

  <h4 class="font-weight-bold spanborder"><span>Education</span></h4>
  <div class="timeline">
    <div class="tl-item">
      <div class="tl-logo"><span class="tl-badge"><i class="fas fa-graduation-cap" style="font-size:1.1rem"></i></span></div>
      <div class="tl-body">
        <div class="tl-head">
          <h5>M.Sc, Electrical Engineering — Smart Information Processing</h5>
          <span class="tl-company">University of Stuttgart</span>
          <span class="tl-meta">Apr 2021 – Oct 2023</span>
        </div>
        <div class="thesis-card">
          <div class="thesis-label">Master's Thesis</div>
          <p class="thesis-title"><a href="https://docs.google.com/presentation/d/1qaHabKWl-QXc8DV3TMOxzGVV6rjurU3M/edit?usp=sharing&amp;ouid=115597177769299411644&amp;rtpof=true&amp;sd=true" target="_blank" rel="noopener">Differential Privacy as a privacy-preserving technique in Federated Learning</a></p>
          <p class="thesis-obj"><strong>Objective:</strong> Distributed training and privacy.</p>
          <p class="thesis-sup"><strong>Supervisors:</strong> <a href="https://www.uni-stuttgart.de/en/press/experts/Prof.-Michael-Weyrich/" target="_blank" rel="noopener">Prof. Michael Weyrich</a>, <a href="https://scholar.google.com/citations?user=pcCoDN4AAAAJ&amp;hl=en" target="_blank" rel="noopener">Baran Can Gül</a></p>
          <ul class="thesis-points">
            <li>Federated Learning decentralises deep-learning training across client devices — only local model weights are sent to a central server for aggregation into a global model, avoiding central data storage and easing server load.</li>
            <li>Exposed model weights are a security risk: an adversary can reconstruct a client's original data using generative models.</li>
            <li>Introduced Differential Privacy at training time via <strong>DP-SGD</strong>, which clips and adds noise to constrain each example's influence — trading some accuracy and convergence time for privacy.</li>
            <li>Used <strong>privacy accounting</strong> (epochs, noise, clipping bound) to quantify privacy, and searched for the optimal noise level that maximises privacy without degrading accuracy — evaluated on driver-profile identification across three client devices.</li>
          </ul>
          <div class="thesis-tags">
            <span>Federated Learning</span><span>Differential Privacy</span><span>DP-SGD</span><span>Privacy Accounting</span><span>Distributed Training</span><span>Noise–Accuracy Trade-off</span><span>Driver Profile Identification</span>
          </div>
        </div>
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-logo"><span class="tl-badge"><i class="fas fa-graduation-cap" style="font-size:1.1rem"></i></span></div>
      <div class="tl-body">
        <div class="tl-head">
          <h5>B.E., Electronics &amp; Communication Engineering</h5>
          <span class="tl-company">AMC Engineering College</span>
          <span class="tl-meta">2013 – 2017</span>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
  (function () {
    if (typeof gtag !== 'function') return;

    var ref = document.referrer;
    var source;

    if (!ref) {
      source = 'direct';
    } else {
      try {
        var refHost = new URL(ref).hostname;
        source = refHost === window.location.hostname ? 'internal' : 'external';
      } catch (e) {
        source = 'unknown';
      }
    }

    if (source === 'direct' || source === 'external') {
      gtag('event', 'about_direct_visit', {
        referrer_type: source,
        referrer: ref || '(none)'
      });
    }
  })();
</script>
