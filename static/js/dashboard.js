/**
 * Tactical Detection Console — upload, webcam capture, /detect API.
 */
(function () {
    'use strict';

    const fileInput      = document.getElementById('fileInput');
    const btnUpload      = document.getElementById('btnUpload');
    const btnWebcam      = document.getElementById('btnWebcam');
    const webcamOverlay  = document.getElementById('webcamOverlay');
    const webcamVideo    = document.getElementById('webcamVideo');
    const btnWebcamCancel  = document.getElementById('btnWebcamCancel');
    const btnWebcamCapture = document.getElementById('btnWebcamCapture');
    const runStatus      = document.getElementById('runStatus');
    const runStatusText  = document.getElementById('runStatusText');
    const metaDetections = document.getElementById('metaDetections');
    const metaInput      = document.getElementById('metaInput');
    const panelOriginal     = document.getElementById('panelOriginal');
    const panelPreprocessed = document.getElementById('panelPreprocessed');
    const panelDetection    = document.getElementById('panelDetection');
    const tableBody      = document.getElementById('detectionTableBody');
    const tableCount     = document.getElementById('tableCount');

    let webcamStream = null;

    /* ── Status helpers ─────────────────────────────────────────── */
    function setStatus(state, text) {
        runStatus.className = 'dashboard__status' + (state ? ' is-' + state : '');
        runStatusText.textContent = text;
    }

    function setProcessing() {
        setStatus('processing', 'Processing…');
        btnUpload.disabled = true;
        btnWebcam.disabled = true;
    }

    function setIdle() {
        setStatus('', 'Idle');
        btnUpload.disabled = false;
        btnWebcam.disabled = false;
    }

    function setDone(count) {
        setStatus('done', 'Done · ' + count + ' detection' + (count !== 1 ? 's' : ''));
        btnUpload.disabled = false;
        btnWebcam.disabled = false;
    }

    function setError(msg) {
        setStatus('error', msg);
        btnUpload.disabled = false;
        btnWebcam.disabled = false;
    }

    /* ── Panel / table rendering ────────────────────────────────── */
    function showImage(panel, b64) {
        panel.innerHTML = '';
        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + b64;
        img.alt = panel.id;
        panel.appendChild(img);
    }

    function renderTable(detections) {
        tableCount.textContent = detections.length + ' object' + (detections.length !== 1 ? 's' : '');

        if (!detections.length) {
            tableBody.innerHTML = '<tr class="empty-row"><td colspan="4">No detections yet.</td></tr>';
            return;
        }

        tableBody.innerHTML = detections.map(function (d, i) {
            const pct = Math.round(d.confidence * 100);
            const bbox = d.bbox.join(', ');
            const idx = String(i + 1).padStart(2, '0');
            return (
                '<tr>' +
                    '<td class="col-idx">' + idx + '</td>' +
                    '<td class="col-object">' + d.label + '</td>' +
                    '<td class="col-confidence">' +
                        '<div class="confidence-bar"><div class="confidence-bar__fill" style="width:' + pct + '%"></div></div>' +
                        '<span class="confidence-pct">' + pct + '%</span>' +
                    '</td>' +
                    '<td class="col-bbox">[' + bbox + ']</td>' +
                '</tr>'
            );
        }).join('');
    }

    function renderResult(data) {
        showImage(panelOriginal,     data.images.original);
        showImage(panelPreprocessed, data.images.preprocessed);
        showImage(panelDetection,    data.images.annotated);
        renderTable(data.detections);

        const m = data.metrics;
        metaDetections.textContent = 'Detections: ' + m.detection_count;
        metaInput.textContent = 'Input: ' + m.image_size;
        setDone(m.detection_count);
    }

    /* ── API call ───────────────────────────────────────────────── */
    async function sendDetect(formData) {
        setProcessing();
        try {
            const res = await fetch('/detect', { method: 'POST', body: formData });
            const data = await res.json();
            if (!res.ok) {
                setError(data.error || 'Detection failed');
                return;
            }
            renderResult(data);
        } catch (err) {
            setError('Network error');
            console.error(err);
        }
    }

    /* ── Upload ─────────────────────────────────────────────────── */
    btnUpload.addEventListener('click', function () {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        const file = fileInput.files[0];
        if (!file) return;
        const fd = new FormData();
        fd.append('image', file);
        sendDetect(fd);
        fileInput.value = '';
    });

    /* ── Webcam (browser-side getUserMedia) ─────────────────────── */
    async function openWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamVideo.srcObject = webcamStream;
            webcamOverlay.classList.add('is-open');
        } catch (err) {
            setError('Webcam access denied');
            console.error(err);
        }
    }

    function closeWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(function (t) { t.stop(); });
            webcamStream = null;
        }
        webcamVideo.srcObject = null;
        webcamOverlay.classList.remove('is-open');
    }

    btnWebcam.addEventListener('click', openWebcam);
    btnWebcamCancel.addEventListener('click', closeWebcam);

    btnWebcamCapture.addEventListener('click', function () {
        const canvas = document.createElement('canvas');
        canvas.width  = webcamVideo.videoWidth;
        canvas.height = webcamVideo.videoHeight;
        canvas.getContext('2d').drawImage(webcamVideo, 0, 0);
        closeWebcam();

        canvas.toBlob(function (blob) {
            if (!blob) { setError('Capture failed'); return; }
            const fd = new FormData();
            fd.append('image', blob, 'webcam.jpg');
            sendDetect(fd);
        }, 'image/jpeg', 0.92);
    });

})();
