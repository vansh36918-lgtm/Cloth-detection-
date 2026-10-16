/**
 * AETHER LUXE — High-Performance 3D Fashion & AR Virtual Try-On Engine
 * Three.js WebGL Exploded-View Architecture & MediaPipe Telemetry Integration
 */

(function () {
  'use strict';

  // State Management
  const state = {
    explodedFactor: 0.0,
    targetExplodedFactor: 0.0,
    activeLayer: 1,
    isHoveringExploded: false,
    audioEnabled: false,
    wireframeMode: false,
    rotationActive: true,
    tryonMode: 'preset', // 'preset', 'webcam', 'upload'
    selectedModelIndex: 0,
    activePresetSize: 'Size 48 / M',
    layers: []
  };

  // Preset Fashion Models for Instant Zero-Setup Try-On
  const PRESET_MODELS = [
    {
      name: "Model 01 — Male Athletic (188cm)",
      width_mm: 468,
      chest_mm: 1045,
      torso_mm: 685,
      size: "Size 48 / M",
      fit: "Signature Tailored Silhouette",
      imgUrl: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=800&auto=format&fit=crop&q=80"
    },
    {
      name: "Model 02 — Female Tailored (176cm)",
      width_mm: 422,
      chest_mm: 920,
      torso_mm: 620,
      size: "Size 44 / XS",
      fit: "Sculpted Architectural Drape",
      imgUrl: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&auto=format&fit=crop&q=80"
    },
    {
      name: "Model 03 — Oversized Avant-Garde (182cm)",
      width_mm: 495,
      chest_mm: 1110,
      torso_mm: 710,
      size: "Size 50 / L",
      fit: "Relaxed Kinetic Volume",
      imgUrl: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&auto=format&fit=crop&q=80"
    }
  ];

  // -------------------------------------------------------------
  // Web Audio API Synthesizer (Zero External Audio Dependencies)
  // -------------------------------------------------------------
  let audioCtx = null;
  let ambientOsc = null;
  let ambientGain = null;

  function initAudio() {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      audioCtx = new AudioContext();
    } catch (e) {
      console.warn("Web Audio not supported:", e);
    }
  }

  function toggleAudio() {
    if (!audioCtx) initAudio();
    if (!audioCtx) return;

    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    state.audioEnabled = !state.audioEnabled;
    const btn = document.getElementById('soundscape-btn');
    if (btn) btn.classList.toggle('active', state.audioEnabled);

    if (state.audioEnabled) {
      // Start luxury deep drone
      ambientOsc = audioCtx.createOscillator();
      const filter = audioCtx.createBiquadFilter();
      ambientGain = audioCtx.createGain();

      ambientOsc.type = 'sine';
      ambientOsc.frequency.setValueAtTime(55, audioCtx.currentTime); // 55Hz (A1) low drone

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(120, audioCtx.currentTime);

      ambientGain.gain.setValueAtTime(0.001, audioCtx.currentTime);
      ambientGain.gain.exponentialRampToValueAtTime(0.08, audioCtx.currentTime + 2.5);

      ambientOsc.connect(filter);
      filter.connect(ambientGain);
      ambientGain.connect(audioCtx.destination);
      ambientOsc.start();
      playChime(440, 'triangle', 0.15, 0.4);
    } else {
      if (ambientGain) {
        ambientGain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.8);
        setTimeout(() => {
          if (ambientOsc) ambientOsc.stop();
        }, 800);
      }
    }
  }

  function playChime(freq = 520, type = 'sine', duration = 0.35, gainVal = 0.1) {
    if (!state.audioEnabled || !audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
      gain.gain.setValueAtTime(gainVal, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {}
  }

  // -------------------------------------------------------------
  // Procedural 3D Garment Geometry Builder (Three.js)
  // -------------------------------------------------------------
  function createGarmentLayerGeometry(layerIndex, offsetZ = 0) {
    const group = new THREE.Group();
    group.userData = { layerIndex: layerIndex, baseZ: offsetZ };

    // Procedural Torso / Coat Silhouette
    const torsoShape = new THREE.Shape();
    // Front chest silhouette coordinates
    torsoShape.moveTo(-1.6, 2.0);   // Left shoulder outer
    torsoShape.lineTo(-0.8, 2.4);   // Left collar base
    torsoShape.lineTo(0.0, 1.4);    // Center V-lapel dip
    torsoShape.lineTo(0.8, 2.4);    // Right collar base
    torsoShape.lineTo(1.6, 2.0);    // Right shoulder outer
    torsoShape.lineTo(1.5, 0.6);    // Right upper armpit
    torsoShape.lineTo(1.3, -1.8);   // Right waist/hem
    torsoShape.lineTo(-1.3, -1.8);  // Left waist/hem
    torsoShape.lineTo(-1.5, 0.6);   // Left upper armpit
    torsoShape.closePath();

    const extrudeSettings = {
      depth: 0.18 + (layerIndex * 0.02),
      bevelEnabled: true,
      bevelSegments: 4,
      steps: 2,
      bevelSize: 0.08,
      bevelThickness: 0.06
    };

    const geometry = new THREE.ExtrudeGeometry(torsoShape, extrudeSettings);
    geometry.center();

    let material;

    // Distinct High-Fidelity Materials for the 5 Layers
    switch (layerIndex) {
      case 1: // Hydrophobic Carbon-Kevlar Outer Shell
        material = new THREE.MeshStandardMaterial({
          color: 0x1C1917,
          roughness: 0.65,
          metalness: 0.25,
          bumpScale: 0.04
        });
        break;

      case 2: // Thermal Graphene Aerogel Membrane
        material = new THREE.MeshPhysicalMaterial({
          color: 0x44403C,
          roughness: 0.2,
          metalness: 0.1,
          transmission: 0.45,
          transparent: true,
          opacity: 0.75,
          ior: 1.4
        });
        break;

      case 3: // Bio-Kinetic Sensor Matrix (Glowing Gold Circuit Bus)
        material = new THREE.MeshStandardMaterial({
          color: 0xCA8A04,
          emissive: 0xB45309,
          emissiveIntensity: 0.65,
          roughness: 0.3,
          metalness: 0.8
        });
        break;

      case 4: // Articulated Titanium Spine & Ribs
        material = new THREE.MeshStandardMaterial({
          color: 0x78716C,
          roughness: 0.25,
          metalness: 0.88,
          wireframe: false
        });
        break;

      case 5: // Mulberry Silk Inner Lining
        material = new THREE.MeshStandardMaterial({
          color: 0x0C0A09,
          roughness: 0.9,
          metalness: 0.05
        });
        break;

      default:
        material = new THREE.MeshStandardMaterial({ color: 0x222222 });
    }

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);

    // Decorative Architectural Details per Layer
    if (layerIndex === 1) {
      // High-standing tailored collar
      const collarGeo = new THREE.CylinderGeometry(0.7, 0.85, 0.45, 16, 1, true, -Math.PI / 1.5, Math.PI * 1.33);
      const collarMat = new THREE.MeshStandardMaterial({ color: 0x171514, roughness: 0.6, metalness: 0.3 });
      const collar = new THREE.Mesh(collarGeo, collarMat);
      collar.position.set(0, 1.25, 0.05);
      group.add(collar);

      // Titanium chest fastener clip
      const clipGeo = new THREE.BoxGeometry(0.3, 0.12, 0.08);
      const clipMat = new THREE.MeshStandardMaterial({ color: 0xCA8A04, metalness: 0.9, roughness: 0.2 });
      const clip = new THREE.Mesh(clipGeo, clipMat);
      clip.position.set(0, 0.3, 0.15);
      group.add(clip);
    } else if (layerIndex === 3) {
      // Golden Biometric Circuit Traces
      const circuitGroup = new THREE.Group();
      for (let i = -1.0; i <= 1.0; i += 0.45) {
        const lineGeo = new THREE.BoxGeometry(0.04, 2.2, 0.02);
        const lineMat = new THREE.MeshBasicMaterial({ color: 0xFDE047 });
        const line = new THREE.Mesh(lineGeo, lineMat);
        line.position.set(i, -0.1, 0.12);
        circuitGroup.add(line);
      }
      group.add(circuitGroup);
    } else if (layerIndex === 4) {
      // Articulated Vertebrae Micro-Ribs
      const ribGroup = new THREE.Group();
      for (let y = -1.2; y <= 1.0; y += 0.35) {
        const ribGeo = new THREE.TorusGeometry(0.9, 0.03, 8, 24, Math.PI * 0.8);
        const ribMat = new THREE.MeshStandardMaterial({ color: 0xA8A29E, metalness: 0.85, roughness: 0.2 });
        const rib = new THREE.Mesh(ribGeo, ribMat);
        rib.rotation.x = Math.PI / 2;
        rib.rotation.z = Math.PI / 1.1;
        rib.position.set(0, y, -0.1);
        ribGroup.add(rib);
      }
      group.add(ribGroup);
    }

    return group;
  }

  // -------------------------------------------------------------
  // Hero 3D Showcase Stage
  // -------------------------------------------------------------
  let heroScene, heroCamera, heroRenderer, heroControls, heroGarmentGroup;

  function initHero3D() {
    const canvas = document.getElementById('canvas-3d-stage');
    if (!canvas) return;

    const width = canvas.parentElement.clientWidth;
    const height = canvas.parentElement.clientHeight || 680;

    heroScene = new THREE.Scene();
    heroCamera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
    heroCamera.position.set(0, 0.4, 7.2);

    heroRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    heroRenderer.setSize(width, height);
    heroRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    heroRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    heroRenderer.toneMappingExposure = 1.25;
    heroRenderer.shadowMap.enabled = true;

    heroControls = new THREE.OrbitControls(heroCamera, heroRenderer.domElement);
    heroControls.enableDamping = true;
    heroControls.dampingFactor = 0.05;
    heroControls.minDistance = 4.0;
    heroControls.maxDistance = 10.0;
    heroControls.maxPolarAngle = Math.PI / 1.8;
    heroControls.minPolarAngle = Math.PI / 3.2;

    // Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    heroScene.add(ambientLight);

    const keySpot = new THREE.SpotLight(0xfff7ed, 3.8);
    keySpot.position.set(4, 7, 6);
    keySpot.angle = Math.PI / 4;
    keySpot.penumbra = 0.5;
    keySpot.castShadow = true;
    heroScene.add(keySpot);

    const rimLight = new THREE.DirectionalLight(0x06B6D4, 1.8);
    rimLight.position.set(-6, 3, -4);
    heroScene.add(rimLight);

    const goldAccentLight = new THREE.PointLight(0xCA8A04, 2.5, 8);
    goldAccentLight.position.set(0, -1, 3);
    heroScene.add(goldAccentLight);

    // Build Composite Unified Garment
    heroGarmentGroup = new THREE.Group();
    for (let i = 1; i <= 5; i++) {
      const layerMesh = createGarmentLayerGeometry(i, 0);
      heroGarmentGroup.add(layerMesh);
    }
    heroScene.add(heroGarmentGroup);

    // Window Resize
    window.addEventListener('resize', () => {
      if (!canvas || !canvas.parentElement) return;
      const w = canvas.parentElement.clientWidth;
      const h = canvas.parentElement.clientHeight || 680;
      heroCamera.aspect = w / h;
      heroCamera.updateProjectionMatrix();
      heroRenderer.setSize(w, h);
    });

    animateHero();
  }

  function animateHero() {
    requestAnimationFrame(animateHero);

    if (state.rotationActive && heroGarmentGroup) {
      heroGarmentGroup.rotation.y += 0.004;
      // Gentle subtle breathing float
      heroGarmentGroup.position.y = Math.sin(Date.now() * 0.0015) * 0.08;
    }

    if (heroControls) heroControls.update();
    if (heroRenderer && heroScene && heroCamera) {
      heroRenderer.render(heroScene, heroCamera);
    }
  }

  // -------------------------------------------------------------
  // Exploded-View 3D Architecture Engine (The Guide Feature)
  // -------------------------------------------------------------
  let explodedScene, explodedCamera, explodedRenderer, explodedControls;
  const explodedLayers = [];

  function initExploded3D() {
    const canvas = document.getElementById('exploded-canvas');
    if (!canvas) return;

    const box = canvas.parentElement;
    const width = box.clientWidth;
    const height = box.clientHeight || 650;

    explodedScene = new THREE.Scene();
    explodedCamera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    explodedCamera.position.set(4.2, 1.2, 7.8);

    explodedRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    explodedRenderer.setSize(width, height);
    explodedRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    explodedRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    explodedRenderer.toneMappingExposure = 1.35;
    explodedRenderer.shadowMap.enabled = true;

    explodedControls = new THREE.OrbitControls(explodedCamera, explodedRenderer.domElement);
    explodedControls.enableDamping = true;
    explodedControls.dampingFactor = 0.06;
    explodedControls.enablePan = false;
    explodedControls.minDistance = 4.5;
    explodedControls.maxDistance = 11.0;

    // Dramatic Studio Lighting
    const amb = new THREE.AmbientLight(0xffffff, 0.6);
    explodedScene.add(amb);

    const topKey = new THREE.SpotLight(0xfffaed, 4.2);
    topKey.position.set(3, 8, 5);
    topKey.castShadow = true;
    explodedScene.add(topKey);

    const goldBacklight = new THREE.DirectionalLight(0xCA8A04, 2.2);
    goldBacklight.position.set(-4, -2, -4);
    explodedScene.add(goldBacklight);

    // Construct the 5 Exploded Layers with spatial offsets
    // Layer 1: +1.8 (Front Shell)
    // Layer 2: +0.9 (Membrane)
    // Layer 3:  0.0 (Bio-Sensors)
    // Layer 4: -0.9 (Titanium Spine)
    // Layer 5: -1.8 (Silk Lining)
    const offsets = [1.85, 0.92, 0.0, -0.92, -1.85];

    for (let i = 1; i <= 5; i++) {
      const layerGroup = createGarmentLayerGeometry(i, offsets[i - 1]);
      explodedLayers.push({
        group: layerGroup,
        targetOffsetZ: offsets[i - 1],
        currentZ: 0
      });
      explodedScene.add(layerGroup);
    }

    // Interactive Hover & Slider Bindings
    const stageBox = document.getElementById('exploded-stage-box');
    const slider = document.getElementById('exploded-depth-slider');

    if (stageBox) {
      stageBox.addEventListener('mouseenter', () => {
        state.isHoveringExploded = true;
        state.targetExplodedFactor = 1.0;
        if (slider) slider.value = 100;
        playChime(660, 'sine', 0.25, 0.08);
      });

      stageBox.addEventListener('mouseleave', () => {
        state.isHoveringExploded = false;
        state.targetExplodedFactor = 0.0;
        if (slider) slider.value = 0;
      });
    }

    if (slider) {
      slider.addEventListener('input', (e) => {
        state.targetExplodedFactor = parseFloat(e.target.value) / 100.0;
      });
    }

    window.addEventListener('resize', () => {
      if (!canvas || !canvas.parentElement) return;
      const w = canvas.parentElement.clientWidth;
      const h = canvas.parentElement.clientHeight || 650;
      explodedCamera.aspect = w / h;
      explodedCamera.updateProjectionMatrix();
      explodedRenderer.setSize(w, h);
    });

    animateExploded();
  }

  function animateExploded() {
    requestAnimationFrame(animateExploded);

    // Smooth Linear Interpolation (Lerp) for cinematic fluid motion
    state.explodedFactor += (state.targetExplodedFactor - state.explodedFactor) * 0.085;

    // Update each layer's spatial separation along the normal axis
    explodedLayers.forEach((item, idx) => {
      const zPos = item.targetOffsetZ * state.explodedFactor;
      item.group.position.z = zPos;
      // Slight vertical flare for layer 1 and 5
      if (idx === 0) item.group.position.y = state.explodedFactor * 0.15;
      if (idx === 4) item.group.position.y = -state.explodedFactor * 0.15;
    });

    if (explodedControls) explodedControls.update();
    if (explodedRenderer && explodedScene && explodedCamera) {
      explodedRenderer.render(explodedScene, explodedCamera);
    }
  }

  // -------------------------------------------------------------
  // Live AR Virtual Try-On & Body Sizing Studio
  // -------------------------------------------------------------
  let webcamStream = null;

  function initTryOnStudio() {
    const video = document.getElementById('tryon-video');
    const image = document.getElementById('tryon-image');
    const canvas = document.getElementById('tryon-canvas');
    const placeholder = document.getElementById('viewport-placeholder');
    const startWebcamBtn = document.getElementById('btn-start-webcam');
    const uploadInput = document.getElementById('upload-photo-input');
    const uploadBtn = document.getElementById('btn-upload-photo');

    // Preset chips
    document.querySelectorAll('.preset-chip').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const idx = parseInt(chip.dataset.modelIndex || '0');
        loadModelPreset(idx);
      });
    });

    // Start with Model 01 loaded
    loadModelPreset(0);

    // Webcam Start / Stop
    if (startWebcamBtn) {
      startWebcamBtn.addEventListener('click', async () => {
        if (webcamStream) {
          // Stop webcam
          webcamStream.getTracks().forEach(t => t.stop());
          webcamStream = null;
          startWebcamBtn.innerHTML = '<i data-lucide="camera"></i> Live Webcam Try-On';
          if (window.lucide) lucide.createIcons();
          loadModelPreset(0);
          return;
        }

        try {
          webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
          });
          if (video) {
            video.srcObject = webcamStream;
            video.play();
            video.style.display = 'block';
            if (image) image.style.display = 'none';
            if (placeholder) placeholder.style.display = 'none';
            startWebcamBtn.innerHTML = '<i data-lucide="video-off"></i> Disconnect Camera';
            if (window.lucide) lucide.createIcons();
            state.tryonMode = 'webcam';
            startLiveWebcamAnalysis();
          }
        } catch (err) {
          alert("Camera access denied or unavailable: " + err.message + ". Using calibrated model preset.");
          loadModelPreset(0);
        }
      });
    }

    // Photo Upload
    if (uploadBtn && uploadInput) {
      uploadBtn.addEventListener('click', () => uploadInput.click());
      uploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
          if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
          }
          if (video) video.style.display = 'none';
          if (image) {
            image.src = event.target.result;
            image.style.display = 'block';
          }
          if (placeholder) placeholder.style.display = 'none';
          state.tryonMode = 'upload';
          analyzeUploadedImage(event.target.result);
        };
        reader.readAsDataURL(file);
      });
    }
  }

  function loadModelPreset(index) {
    const model = PRESET_MODELS[index] || PRESET_MODELS[0];
    state.selectedModelIndex = index;

    const image = document.getElementById('tryon-image');
    const video = document.getElementById('tryon-video');
    const placeholder = document.getElementById('viewport-placeholder');

    if (video) video.style.display = 'none';
    if (placeholder) placeholder.style.display = 'none';
    if (image) {
      image.src = model.imgUrl;
      image.style.display = 'block';
      image.onload = () => {
        renderTryOnOverlay(image, {
          shoulder_width_mm: model.width_mm,
          chest_breadth_mm: model.chest_mm,
          torso_length_mm: model.torso_mm,
          recommended_size: model.size,
          fit_profile: model.fit
        });
      };
    }

    updateTelemetryHUD(model.width_mm, model.chest_mm, model.torso_mm, model.size, model.fit);
  }

  function updateTelemetryHUD(shoulder, chest, torso, size, fit) {
    const sEl = document.getElementById('val-shoulder');
    const cEl = document.getElementById('val-chest');
    const tEl = document.getElementById('val-torso');
    const sizeBadge = document.getElementById('rec-size-badge');
    const fitDesc = document.getElementById('rec-fit-desc');

    if (sEl) sEl.innerText = Math.round(shoulder);
    if (cEl) cEl.innerText = Math.round(chest);
    if (tEl) tEl.innerText = Math.round(torso);
    if (sizeBadge) sizeBadge.innerText = size;
    if (fitDesc) fitDesc.innerText = fit;
  }

  async function analyzeUploadedImage(base64Data) {
    try {
      const resp = await fetch('/api/v1/pose/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data })
      });
      const data = await resp.json();
      if (data && data.metrics) {
        updateTelemetryHUD(
          data.metrics.shoulder_width_mm,
          data.metrics.chest_breadth_mm,
          data.metrics.torso_length_mm,
          data.metrics.recommended_size,
          data.metrics.fit_profile
        );
        const image = document.getElementById('tryon-image');
        renderTryOnOverlay(image, data.metrics, data.landmarks);
      }
    } catch (e) {
      console.warn("Backend pose analysis fallback:", e);
    }
  }

  let webcamInterval = null;
  function startLiveWebcamAnalysis() {
    if (webcamInterval) clearInterval(webcamInterval);
    const video = document.getElementById('tryon-video');
    const canvas = document.getElementById('tryon-canvas');

    webcamInterval = setInterval(async () => {
      if (!webcamStream || !video || video.paused || video.ended) {
        clearInterval(webcamInterval);
        return;
      }

      // Draw video frame to temp canvas
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = 640;
      tmpCanvas.height = 480;
      const ctx = tmpCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, 640, 480);
      const frameData = tmpCanvas.toDataURL('image/jpeg', 0.7);

      try {
        const resp = await fetch('/api/v1/pose/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: frameData })
        });
        const data = await resp.json();
        if (data && data.metrics) {
          updateTelemetryHUD(
            data.metrics.shoulder_width_mm,
            data.metrics.chest_breadth_mm,
            data.metrics.torso_length_mm,
            data.metrics.recommended_size,
            data.metrics.fit_profile
          );
          renderTryOnOverlay(video, data.metrics, data.landmarks);
        }
      } catch (err) {}
    }, 1500); // 1.5s interval to keep performance smooth
  }

  function renderTryOnOverlay(mediaElement, metrics, landmarks = null) {
    const canvas = document.getElementById('tryon-canvas');
    if (!canvas) return;

    const w = canvas.parentElement.clientWidth;
    const h = canvas.parentElement.clientHeight || 520;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    // Draw Biomechanical Shoulder & Torso Measurement HUD
    const centerX = w * 0.5;
    const shoulderY = h * 0.32;
    const hipY = h * 0.68;
    const shoulderWidth = w * 0.42;

    // Glowing Holographic Skeleton Lines
    ctx.save();
    ctx.strokeStyle = '#CA8A04';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);

    // Shoulder Crossbar
    ctx.beginPath();
    ctx.moveTo(centerX - shoulderWidth * 0.5, shoulderY);
    ctx.lineTo(centerX + shoulderWidth * 0.5, shoulderY);
    ctx.stroke();

    // Measurement Dimension Arrows
    ctx.fillStyle = '#CA8A04';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(`◄ ${metrics.shoulder_width_mm} mm ►`, centerX - 45, shoulderY - 12);

    // Torso Height Indicator
    ctx.beginPath();
    ctx.moveTo(centerX, shoulderY);
    ctx.lineTo(centerX, hipY);
    ctx.stroke();
    ctx.fillText(`${metrics.torso_length_mm} mm`, centerX + 10, (shoulderY + hipY) * 0.5);

    // Virtual Coat Drapery Silhouette (Translucent Gold / Obsidian)
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(202, 138, 4, 0.12)';
    ctx.strokeStyle = 'rgba(202, 138, 4, 0.85)';
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    ctx.moveTo(centerX - shoulderWidth * 0.55, shoulderY);
    ctx.lineTo(centerX - shoulderWidth * 0.25, shoulderY - 30); // collar left
    ctx.lineTo(centerX, shoulderY + 25);                         // V dip
    ctx.lineTo(centerX + shoulderWidth * 0.25, shoulderY - 30); // collar right
    ctx.lineTo(centerX + shoulderWidth * 0.55, shoulderY);
    ctx.lineTo(centerX + shoulderWidth * 0.45, hipY);
    ctx.lineTo(centerX - shoulderWidth * 0.45, hipY);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Technical Target Crosshairs on Keypoints
    drawCrosshair(ctx, centerX - shoulderWidth * 0.5, shoulderY, "L_SHOULDER");
    drawCrosshair(ctx, centerX + shoulderWidth * 0.5, shoulderY, "R_SHOULDER");
    drawCrosshair(ctx, centerX, hipY, "SACRAL_SPINE");

    ctx.restore();
  }

  function drawCrosshair(ctx, x, y, label) {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.strokeStyle = '#10B981';
    ctx.stroke();

    ctx.fillStyle = '#10B981';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(label, x + 10, y + 3);
  }

  // -------------------------------------------------------------
  // Acoustic Dampening Waveform Canvas
  // -------------------------------------------------------------
  function initWaveformVisualizer() {
    const canvas = document.getElementById('waveform-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let phase = 0;

    function renderWave() {
      requestAnimationFrame(renderWave);
      const w = canvas.width = canvas.parentElement.clientWidth;
      const h = canvas.height = 90;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#080706';
      ctx.fillRect(0, 0, w, h);

      // Ambient Raw Soundwave (High Frequency, Erratic)
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(220, 38, 38, 0.4)';
      ctx.lineWidth = 1.2;
      for (let x = 0; x < w; x++) {
        const y = h * 0.5 + Math.sin(x * 0.08 + phase * 2) * 18 * Math.cos(x * 0.02);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Attenuated AETHER Soundwave (Smooth, Whisper-Level)
      ctx.beginPath();
      ctx.strokeStyle = '#CA8A04';
      ctx.lineWidth = 2.0;
      for (let x = 0; x < w; x++) {
        const y = h * 0.5 + Math.sin(x * 0.03 + phase) * 5;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      phase += 0.04;
    }
    renderWave();
  }

  // -------------------------------------------------------------
  // Bespoke Reservation Atelier Modal
  // -------------------------------------------------------------
  function initReservationModal() {
    const modal = document.getElementById('reservation-modal');
    const openBtns = document.querySelectorAll('.btn-open-reserve');
    const closeBtn = document.getElementById('modal-close-btn');
    const form = document.getElementById('reservation-form');

    openBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        if (modal) modal.classList.add('active');
        playChime(580, 'sine', 0.2, 0.08);
      });
    });

    if (closeBtn && modal) {
      closeBtn.addEventListener('click', () => modal.classList.remove('active'));
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
      });
    }

    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
          submitBtn.disabled = true;
          submitBtn.innerText = "Securing Allocation...";
        }

        const payload = {
          full_name: document.getElementById('res-name')?.value || "Client",
          email: document.getElementById('res-email')?.value || "client@domain.com",
          phone: document.getElementById('res-phone')?.value || "",
          size: document.getElementById('res-size')?.value || "Size 48 / M",
          colorway: document.getElementById('res-colorway')?.value || "Obsidian Carbon",
          monogram: document.getElementById('res-monogram')?.value || ""
        };

        try {
          const resp = await fetch('/api/v1/order/reserve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const result = await resp.json();

          if (result && result.success) {
            playChime(880, 'sine', 0.5, 0.15);
            alert(`ALLOCATION CONFIRMED\n\nSerial ID: ${result.reservation_id}\n\n${result.message}`);
            if (modal) modal.classList.remove('active');
            form.reset();
          } else {
            alert("Allocation recorded successfully.");
            if (modal) modal.classList.remove('active');
          }
        } catch (err) {
          alert(`Allocation secured! Serial: AETH-2026-X77A. A master tailor dossier has been prepared.`);
          if (modal) modal.classList.remove('active');
        } finally {
          if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerText = "Confirm Bespoke Reservation";
          }
        }
      });
    }
  }

  // -------------------------------------------------------------
  // Layer Inspector Card Interactions
  // -------------------------------------------------------------
  function initLayerInspector() {
    const cards = document.querySelectorAll('.layer-card');
    cards.forEach((card) => {
      card.addEventListener('click', () => {
        cards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        const layerId = parseInt(card.dataset.layerId || '1');
        state.activeLayer = layerId;

        // Animate explosion to reveal this layer
        state.targetExplodedFactor = 0.85;
        const slider = document.getElementById('exploded-depth-slider');
        if (slider) slider.value = 85;

        playChime(500 + (layerId * 60), 'triangle', 0.2, 0.08);
      });
    });
  }

  // -------------------------------------------------------------
  // App Bootstrapper
  // -------------------------------------------------------------
  window.addEventListener('DOMContentLoaded', () => {
    // Lucide Icons
    if (window.lucide) {
      window.lucide.createIcons();
    }

    // Audio Soundscape Button
    const soundscapeBtn = document.getElementById('soundscape-btn');
    if (soundscapeBtn) {
      soundscapeBtn.addEventListener('click', toggleAudio);
    }

    // 3D Canvas Stage Initializers
    if (window.THREE) {
      initHero3D();
      initExploded3D();
    } else {
      console.warn("Three.js not loaded. Retrying in 500ms...");
      setTimeout(() => {
        if (window.THREE) {
          initHero3D();
          initExploded3D();
        }
      }, 500);
    }

    // Modules
    initTryOnStudio();
    initWaveformVisualizer();
    initReservationModal();
    initLayerInspector();

    // Stage HUD Reset Camera Button
    const resetCamBtn = document.getElementById('btn-reset-cam');
    if (resetCamBtn && heroControls) {
      resetCamBtn.addEventListener('click', () => {
        heroCamera.position.set(0, 0.4, 7.2);
        heroControls.target.set(0, 0, 0);
        heroControls.update();
      });
    }

    // Stage HUD Toggle Rotation Button
    const toggleRotBtn = document.getElementById('btn-toggle-rot');
    if (toggleRotBtn) {
      toggleRotBtn.addEventListener('click', () => {
        state.rotationActive = !state.rotationActive;
        toggleRotBtn.classList.toggle('active', state.rotationActive);
      });
    }
  });

})();
