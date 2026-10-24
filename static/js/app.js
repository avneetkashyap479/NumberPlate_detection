document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.upload-form').forEach((form) => {
    const input = form.querySelector('input[type=file]');
    const selected = form.querySelector('.selected-file');
    const zone = form.querySelector('.drop-zone');
    input.addEventListener('change', () => { selected.textContent = input.files[0] ? `${input.files[0].name} · ${(input.files[0].size / 1048576).toFixed(2)} MB` : ''; });
    zone.addEventListener('dragover', (event) => { event.preventDefault(); zone.classList.add('dragging'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
    zone.addEventListener('drop', (event) => { event.preventDefault(); zone.classList.remove('dragging'); if (event.dataTransfer.files.length) { input.files = event.dataTransfer.files; input.dispatchEvent(new Event('change')); } });
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!input.files.length) return;
      const button = form.querySelector('button[type=submit]');
      const area = document.getElementById('result-area');
      button.disabled = true;
      button.innerHTML = '<span class="spinner-border spinner-border-sm"></span><span>Processing locally…</span>';
      area.innerHTML = '<div class="loading-state"><span class="loader"></span><strong>Processing upload</strong><small>Inference runs synchronously. Keep this page open.</small></div>';
      try {
        const response = await fetch(form.action, { method: 'POST', body: new FormData(form) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed.');
        const output = form.dataset.kind === 'image' ? `<img class="media-preview" src="${data.annotated_url}" alt="Annotated result">` : `<video class="media-preview" controls src="${data.output_url}"></video>`;
        const cards = data.detections.map((item) => `<article class="detection-card"><img src="${item.crop_url}" alt="Plate crop"><div><strong>${escapeHtml(item.plate_text || 'Text not recognized')}</strong><small>Raw OCR: ${escapeHtml(item.raw_ocr_text || '—')}</small><small>Detection ${item.detection_confidence.toFixed(3)} · OCR ${item.ocr_confidence == null ? '—' : item.ocr_confidence.toFixed(3)}</small>${item.frame_number ? `<small>Frame ${item.frame_number} · ${item.timestamp_seconds.toFixed(2)} s</small>` : ''}</div></article>`).join('');
        area.innerHTML = `${output}<div class="result-summary"><strong>${data.detections.length} detection event(s)</strong><span>${data.processing_seconds.toFixed(2)} s</span><a class="text-link" href="/job/${data.job_id}">Open saved job ↗</a></div>${cards || '<div class="notice">No plates detected in this input. This is a valid result, not an OCR failure.</div>'}`;
      } catch (error) { area.innerHTML = `<div class="notice error-notice"><strong>Processing did not complete</strong><p>${escapeHtml(error.message)}</p><a href="/history" class="text-link">View recorded job history ↗</a></div>`; }
      finally { button.disabled = false; button.innerHTML = `<span>${form.dataset.kind === 'image' ? 'Run plate detection' : 'Process sampled frames'}</span><span>→</span>`; }
    });
  });
  document.querySelectorAll('.delete-job').forEach((button) => button.addEventListener('click', async () => {
    if (!confirm('Delete this processing record and its managed files?')) return;
    const response = await fetch(`/api/history/${button.dataset.id}`, { method: 'DELETE' });
    const data = await response.json();
    if (response.ok) button.closest('tr').remove(); else alert(data.error || 'Could not delete this record.');
  }));
  const colors = { text: '#eaf4f2', muted: '#91a5a4', grid: 'rgba(157,190,185,.12)', mint: '#54e0ba', cyan: '#59bfe0' };
  if (window.platewatchDaily?.length && document.getElementById('activityChart')) {
    const rows = window.platewatchDaily;
    new Chart(document.getElementById('activityChart'), { type: 'bar', data: { labels: rows.map((x) => x.day), datasets: [{ label: 'Jobs', data: rows.map((x) => x.jobs), backgroundColor: colors.mint, borderRadius: 3 }, { label: 'Successful', data: rows.map((x) => x.successful), backgroundColor: colors.cyan, borderRadius: 3 }] }, options: chartOptions(colors) });
  }
  if (window.platewatchConfidence?.length && document.getElementById('confidenceChart')) {
    const rows = window.platewatchConfidence;
    new Chart(document.getElementById('confidenceChart'), { type: 'doughnut', data: { labels: rows.map((x) => x.bucket), datasets: [{ data: rows.map((x) => x.count), backgroundColor: [colors.mint, colors.cyan, '#f1bb6b', '#e77b70', '#a1a9c9'], borderWidth: 0 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: colors.muted, padding: 18 } } } } });
  }
});
function chartOptions(colors) { return { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: colors.muted, usePointStyle: true, boxWidth: 7 } } }, scales: { x: { ticks: { color: colors.muted }, grid: { display: false } }, y: { beginAtZero: true, ticks: { precision: 0, color: colors.muted }, grid: { color: colors.grid } } } }; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
