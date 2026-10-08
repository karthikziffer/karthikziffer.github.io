/*
  Adds an Expand button to every figure.media-frame in a post and opens
  the image or GIF in a full-screen viewer. Esc, the close button, or a
  click outside the image closes it.
*/
(function () {
  var figures = document.querySelectorAll('.article-post figure.media-frame');
  if (!figures.length) return;

  var dialog = document.createElement('dialog');
  dialog.className = 'media-lightbox';
  dialog.innerHTML =
    '<button type="button" class="media-lightbox-close" aria-label="Close">&times;</button>' +
    '<img alt="">' +
    '<p class="media-lightbox-caption"></p>';
  document.body.appendChild(dialog);

  var bigImage = dialog.querySelector('img');
  var caption = dialog.querySelector('.media-lightbox-caption');

  function open(img, figure) {
    if (typeof dialog.showModal !== 'function') {
      window.open(img.currentSrc || img.src, '_blank');
      return;
    }
    // Setting src again restarts a GIF from its first frame.
    bigImage.src = img.currentSrc || img.src;
    bigImage.alt = img.alt;
    if (img.naturalWidth && img.naturalHeight) {
      bigImage.style.setProperty('--ar', img.naturalWidth / img.naturalHeight);
    }
    var figcaption = figure.querySelector('figcaption');
    caption.textContent = figcaption ? figcaption.textContent : '';
    dialog.showModal();
  }

  dialog.addEventListener('click', function (event) {
    if (event.target === dialog || event.target.closest('.media-lightbox-close')) {
      dialog.close();
    }
  });

  dialog.addEventListener('close', function () {
    bigImage.removeAttribute('src');
  });

  var expandIcon =
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">' +
    '<path d="M10 2h4v4M6 14H2v-4M14 2 9.5 6.5M2 14l4.5-4.5"/></svg>';

  figures.forEach(function (figure) {
    var img = figure.querySelector('img');
    if (!img) return;

    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'media-expand';
    button.setAttribute('aria-label', 'Expand: ' + (img.alt || 'image'));
    button.innerHTML = expandIcon + '<span>Expand</span>';
    (figure.querySelector('.media-frame-bar') || figure).appendChild(button);

    img.classList.add('is-expandable');
    button.addEventListener('click', function () { open(img, figure); });
    img.addEventListener('click', function () { open(img, figure); });
  });
})();
