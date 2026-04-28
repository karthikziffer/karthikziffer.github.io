---
title: "Projects"
layout: page
permalink: "/projects.html"
---

<ul class="list-unstyled" id="gh-projects-list">
  <li class="text-muted"><small>Loading projects from GitHub…</small></li>
</ul>

<script>
(function () {
  var listEl = document.getElementById('gh-projects-list');
  if (!listEl) return;

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmtDate(iso) {
    var d = new Date(iso);
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  }

  var url = 'https://api.github.com/users/karthikziffer/repos?per_page=100&sort=updated&type=owner';
  fetch(url)
    .then(function (r) { return r.json(); })
    .then(function (repos) {
      var filtered = (repos || []).filter(function (r) {
        return !r.fork && r.name !== 'karthikziffer.github.io';
      }).sort(function (a, b) {
        return new Date(b.pushed_at) - new Date(a.pushed_at);
      });
      if (filtered.length === 0) {
        listEl.innerHTML = '<li class="text-muted"><small>No projects found.</small></li>';
        return;
      }
      listEl.innerHTML = filtered.map(function (r) {
        var name = escapeHtml(r.name);
        var desc = r.description ? escapeHtml(r.description) : '';
        var lang = r.language ? escapeHtml(r.language) : '';
        var meta = [];
        if (lang) meta.push(lang);
        meta.push('Updated ' + fmtDate(r.pushed_at));
        return ''
          + '<li class="mb-4">'
          + '<h5 class="mb-1 font-weight-bold">'
          + '<a class="text-dark" href="' + r.html_url + '" target="_blank" rel="noopener">' + name + '</a>'
          + '</h5>'
          + (desc ? '<p class="mb-1">' + desc + '</p>' : '')
          + '<small class="text-muted">' + meta.join(' · ') + '</small>'
          + '</li>';
      }).join('');
    })
    .catch(function () {
      listEl.innerHTML = '<li class="text-muted"><small>Could not load projects from GitHub.</small></li>';
    });
})();
</script>
