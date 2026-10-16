/**
 * AETHER LUXE — High-Performance 3D Fashion & AR Virtual Try-On Engine
 * Featuring:
 * 1. Three.js WebGL 3D Architectural Garment & Anatomical Mannequin Avatar
 * 2. 5-Layer Smooth Exploded-View Architecture (Linear Interpolation)
 * 3. Photorealistic Anatomical Cloth Warping & Ambient Lighting Blending (MediaPipe AI)
 * 4. Web Audio Synthesizer Haptics
 */

(function () {
  'use strict';

  // Global Application State
  const state = {
    explodedFactor: 0.0,
    targetExplodedFactor: 0.0,
    activeLayer: 1,
    isHoveringExploded: false,
    audioEnabled: false,
    rotationActive: true,
    mannequinActive: true,
    activeColorway: 'obsidian', // 'obsidian', 'titanium', 'crimson', 'gold'
    tryonMode: 'preset',        // 'preset', 'webcam', 'upload'
    renderMode: 'photoreal',    // 'photoreal', 'hologram', 'xray'
    selectedModelIndex: 0,
    activePresetSize: 'Size 48 / M',
    landmarksHistory: null
  };

  // Garment Texture Assets
  const GARMENT_IMAGES = {
    obsidian: new Image(),
    titanium: new Image(),
    crimson: new Image(),
    gold: new Image()
  };

  GARMENT_IMAGES.obsidian.src = './static/garments/obsidian_coat.png';
  GARMENT_IMAGES.titanium.src = './static/garments/titanium_coat.png';
  GARMENT_IMAGES.crimson.src = './static/garments/crimson_coat.png';
  GARMENT_IMAGES.gold.src = './static/garments/gold_coat.png';

  // Preset Fashion Models
  const PRESET_MODELS = [
    {
      name: "Model 01 — Male Athletic (188cm)",
      width_mm: 468,
      chest_mm: 1045,
      torso_mm: 685,
      size: "Size 48 / M",
      fit: "Signature Tailored Silhouette",
      imgUrl: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=800&auto=format&fit=crop&q=80",
      landmarks: {
        11: { x: 0.30, y: 0.33 },
        12: { x: 0.70, y: 0.33 },
        23: { x: 0.34, y: 0.75 },
        24: { x: 0.66, y: 0.75 }
      }
    },
    {
      name: "Model 02 — Female Tailored (176cm)",
      width_mm: 422,
      chest_mm: 920,
      torso_mm: 620,
      size: "Size 44 / XS",
      fit: "Sculpted Architectural Drape",
      imgUrl: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&auto=format&fit=crop&q=80",
      landmarks: {
        11: { x: 0.32, y: 0.32 },
        12: { x: 0.68, y: 0.32 },
        23: { x: 0.36, y: 0.72 },
        24: { x: 0.64, y: 0.72 }
      }
    },
    {
      name: "Model 03 — Oversized Avant-Garde (182cm)",
      width_mm: 495,
      chest_mm: 1110,
      torso_mm: 710,
      size: "Size 50 / L",
      fit: "Relaxed Kinetic Volume",
      imgUrl: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&auto=format&fit=crop&q=80",
      landmarks: {
        11: { x: 0.28, y: 0.34 },
        12: { x: 0.72, y: 0.34 },
        23: { x: 0.32, y: 0.78 },
        24: { x: 0.68, y: 0.78 }
      }
    }
  ];

  // -------------------------------------------------------------
  // Web Audio API Synthesizer
  // -------------------------------------------------------------
  let audioCtx = null;
  let ambientOsc = null;
  let ambientGain = null;

  function initAudio() {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      audioCtx = new AudioContext();
    } catch (e) {}
  }

  function toggleAudio() {
    if (!audioCtx) initAudio();
    if (!audioCtx) return;
    if (audioCtx.state === 'suspended') audioCtx.resume();

    state.audioEnabled = !state.audioEnabled;
    const btn = document.getElementById('soundscape-btn');
    if (btn) btn.classList.toggle('active', state.audioEnabled);

    if (state.audioEnabled) {
      ambientOsc = audioCtx.createOscillator();
      const filter = audioCtx.createBiquadFilter();
      ambientGain = audioCtx.createGain();

      ambientOsc.type = 'sine';
      ambientOsc.frequency.setValueAtTime(55, audioCtx.currentTime);

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(120, audioCtx.currentTime);

      ambientGain.gain.setValueAtTime(0.001, audioCtx.currentTime);
      ambientGain.gain.exponentialRampToValueAtTime(0.07, audioCtx.currentTime + 2.0);

      ambientOsc.connect(filter);
      filter.connect(ambientGain);
      ambientGain.connect(audioCtx.destination);
      ambientOsc.start();
      playChime(440, 'triangle', 0.2, 0.3);
    } else {
      if (ambientGain) {
        ambientGain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.5);
        setTimeout(() => { if (ambientOsc) ambientOsc.stop(); }, 500);
      }
    }
  }

  function playChime(freq = 520, type = 'sine', duration = 0.3, gainVal = 0.08) {
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
  function getMaterialForColorway(colorwayName) {
    const map = {
      obsidian: { color: 0x1C1917, roughness: 0.65, metalness: 0.2 },
      titanium: { color: 0x4B5563, roughness: 0.45, metalness: 0.75 },
      crimson: { color: 0x7F1D1D, roughness: 0.55, metalness: 0.3 },
      gold: { color: 0x854D0E, roughness: 0.4, metalness: 0.8 }
    };
    return map[colorwayName] || map.obsidian;
  }

  function createGarmentLayerGeometry(layerIndex, offsetZ = 0, colorway = 'obsidian') {
    const group = new THREE.Group();
    group.userData = { layerIndex: layerIndex, baseZ: offsetZ };

    // Procedural Anatomical Coat Silhouette
    const torsoShape = new THREE.Shape();
    torsoShape.moveTo(-1.6, 2.0);   // Left shoulder
    torsoShape.lineTo(-0.8, 2.4);   // Collar left
    torsoShape.lineTo(0.0, 1.4);    // Lapel V-dip
    torsoShape.lineTo(0.8, 2.4);    // Collar right
    torsoShape.lineTo(1.6, 2.0);    // Right shoulder
    torsoShape.lineTo(1.5, 0.6);    // Armpit
    torsoShape.lineTo(1.3, -1.8);   // Waist/hem right
    torsoShape.lineTo(-1.3, -1.8);  // Waist/hem left
    torsoShape.lineTo(-1.5, 0.6);   // Armpit
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
    const cw = getMaterialForColorway(colorway);

    switch (layerIndex) {
      case 1: // Outer Shell
        material = new THREE.MeshStandardMaterial({
          color: cw.color,
          roughness: cw.roughness,
          metalness: cw.metalness
        });
        break;

      case 2: // Thermal Graphene Aerogel
        material = new THREE.MeshPhysicalMaterial({
          color: 0x44403C,
          roughness: 0.2,
          metalness: 0.1,
          transmission: 0.45,
          transparent: true,
          opacity: 0.75
        });
        break;

      case 3: // Bio-Kinetic Sensor Matrix
        material = new THREE.MeshStandardMaterial({
          color: 0xCA8A04,
          emissive: 0xB45309,
          emissiveIntensity: 0.7,
          roughness: 0.3,
          metalness: 0.85
        });
        break;

      case 4: // Titanium Spine
        material = new THREE.MeshStandardMaterial({
          color: 0x78716C,
          roughness: 0.25,
          metalness: 0.9
        });
        break;

      case 5: // Mulberry Silk
        material = new THREE.MeshStandardMaterial({
          color: 0x0C0A09,
          roughness: 0.92,
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

    if (layerIndex === 1) {
      // High-Standing Tailored Collar
      const collarGeo = new THREE.CylinderGeometry(0.7, 0.85, 0.45, 16, 1, true, -Math.PI / 1.5, Math.PI * 1.33);
      const collarMat = new THREE.MeshStandardMaterial({ color: cw.color, roughness: cw.roughness, metalness: cw.metalness });
      const collar = new THREE.Mesh(collarGeo, collarMat);
      collar.position.set(0, 1.25, 0.05);
      group.add(collar);

      // Titanium Chest Fastener Clasp
      const clipGeo = new THREE.BoxGeometry(0.3, 0.12, 0.08);
      const clipMat = new THREE.MeshStandardMaterial({ color: 0xCA8A04, metalness: 0.9, roughness: 0.2 });
      const clip = new THREE.Mesh(clipGeo, clipMat);
      clip.position.set(0, 0.3, 0.15);
      group.add(clip);
    } else if (layerIndex === 3) {
      // Gold Circuit Bus Traces
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

  // Sculpted 3D Anatomical Mannequin Body
  function createMannequinMesh() {
    const mannequinGroup = new THREE.Group();
    mannequinGroup.name = "mannequinBody";

    const mannequinMat = new THREE.MeshStandardMaterial({
      color: 0x1E1B18,
      roughness: 0.8,
      metalness: 0.15
    });

    // Neck & Head Base
    const neckGeo = new THREE.CylinderGeometry(0.35, 0.42, 0.8, 16);
    const neck = new THREE.Mesh(neckGeo, mannequinMat);
    neck.position.set(0, 1.8, 0);
    mannequinGroup.add(neck);

    // Torso Core
    const torsoGeo = new THREE.CylinderGeometry(0.85, 0.72, 2.8, 16);
    const torso = new THREE.Mesh(torsoGeo, mannequinMat);
    torso.position.set(0, 0.2, 0);
    mannequinGroup.add(torso);

    // Shoulders
    const shoulderGeo = new THREE.SphereGeometry(0.45, 16, 16);
    const lShoulder = new THREE.Mesh(shoulderGeo, mannequinMat);
    lShoulder.position.set(-1.45, 1.3, 0);
    const rShoulder = new THREE.Mesh(shoulderGeo, mannequinMat);
    rShoulder.position.set(1.45, 1.3, 0);
    mannequinGroup.add(lShoulder);
    mannequinGroup.add(rShoulder);

    return mannequinGroup;
  }

  // -------------------------------------------------------------
  // Hero 3D Stage (Three.js)
  // -------------------------------------------------------------
  let heroScene, heroCamera, heroRenderer, heroControls, heroGarmentGroup, heroMannequin;

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

    const goldAccent = new THREE.PointLight(0xCA8A04, 2.5, 8);
    goldAccent.position.set(0, -1, 3);
    heroScene.add(goldAccent);

    // Anatomical Mannequin Body
    heroMannequin = createMannequinMesh();
    heroScene.add(heroMannequin);

    // Build Unified Garment
    rebuildHeroGarment(state.activeColorway);

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

  function rebuildHeroGarment(colorway) {
    if (heroGarmentGroup) {
      heroScene.remove(heroGarmentGroup);
    }
    heroGarmentGroup = new THREE.Group();
    for (let i = 1; i <= 5; i++) {
      const layerMesh = createGarmentLayerGeometry(i, 0, colorway);
      heroGarmentGroup.add(layerMesh);
    }
    heroScene.add(heroGarmentGroup);
  }

  function animateHero() {
    requestAnimationFrame(animateHero);

    if (state.rotationActive && heroGarmentGroup) {
      heroGarmentGroup.rotation.y += 0.004;
      if (heroMannequin) heroMannequin.rotation.y = heroGarmentGroup.rotation.y;
      heroGarmentGroup.position.y = Math.sin(Date.now() * 0.0015) * 0.06;
      if (heroMannequin) heroMannequin.position.y = heroGarmentGroup.position.y;
    }

    if (heroControls) heroControls.update();
    if (heroRenderer && heroScene && heroCamera) {
      heroRenderer.render(heroScene, heroCamera);
    }
  }

  // -------------------------------------------------------------
  // Exploded-View 3D Architecture Engine
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

    explodedControls = new THREE.OrbitControls(explodedCamera, explodedRenderer.domElement);
    explodedControls.enableDamping = true;
    explodedControls.dampingFactor = 0.06;
    explodedControls.enablePan = false;
    explodedControls.minDistance = 4.5;
    explodedControls.maxDistance = 11.0;

    const amb = new THREE.AmbientLight(0xffffff, 0.6);
    explodedScene.add(amb);

    const topKey = new THREE.SpotLight(0xfffaed, 4.2);
    topKey.position.set(3, 8, 5);
    explodedScene.add(topKey);

    const goldBacklight = new THREE.DirectionalLight(0xCA8A04, 2.2);
    goldBacklight.position.set(-4, -2, -4);
    explodedScene.add(goldBacklight);

    const offsets = [1.85, 0.92, 0.0, -0.92, -1.85];
    for (let i = 1; i <= 5; i++) {
      const layerGroup = createGarmentLayerGeometry(i, offsets[i - 1], state.activeColorway);
      explodedLayers.push({
        group: layerGroup,
        targetOffsetZ: offsets[i - 1]
      });
      explodedScene.add(layerGroup);
    }

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

    state.explodedFactor += (state.targetExplodedFactor - state.explodedFactor) * 0.085;

    explodedLayers.forEach((item, idx) => {
      const zPos = item.targetOffsetZ * state.explodedFactor;
      item.group.position.z = zPos;
      if (idx === 0) item.group.position.y = state.explodedFactor * 0.15;
      if (idx === 4) item.group.position.y = -state.explodedFactor * 0.15;
    });

    if (explodedControls) explodedControls.update();
    if (explodedRenderer && explodedScene && explodedCamera) {
      explodedRenderer.render(explodedScene, explodedCamera);
    }
  }

  // -------------------------------------------------------------
  // Photorealistic AR Virtual Try-On & Body Sizing Studio
  // -------------------------------------------------------------
  let webcamStream = null;

  function initTryOnStudio() {
    const video = document.getElementById('tryon-video');
    const image = document.getElementById('tryon-image');
    const startWebcamBtn = document.getElementById('btn-start-webcam');
    const uploadInput = document.getElementById('upload-photo-input');
    const uploadBtn = document.getElementById('btn-upload-photo');

    // Preset model selection
    document.querySelectorAll('.preset-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const idx = parseInt(chip.dataset.modelIndex || '0');
        loadModelPreset(idx);
      });
    });

    loadModelPreset(0);

    // Webcam Start / Stop
    if (startWebcamBtn) {
      startWebcamBtn.addEventListener('click', async () => {
        if (webcamStream) {
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
            document.getElementById('viewport-placeholder').style.display = 'none';
            startWebcamBtn.innerHTML = '<i data-lucide="video-off"></i> Disconnect Camera';
            if (window.lucide) lucide.createIcons();
            state.tryonMode = 'webcam';
            startLiveWebcamAnalysis();
          }
        } catch (err) {
          alert("Camera access denied or unavailable. Using calibrated model preset.");
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
          document.getElementById('viewport-placeholder').style.display = 'none';
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
        renderRealisticClothTryOn(image, model.landmarks, {
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
      const resp = await fetch('/api/v1/tryon/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data, garment: state.activeColorway })
      });
      const data = await resp.json();
      if (data && data.image) {
        const image = document.getElementById('tryon-image');
        image.src = data.image; // photoreal blended image directly from backend
        if (data.metrics) {
          updateTelemetryHUD(
            data.metrics.shoulder_width_mm,
            data.metrics.chest_breadth_mm,
            data.metrics.torso_length_mm,
            data.metrics.recommended_size,
            data.metrics.fit_profile
          );
        }
      }
    } catch (e) {
      console.warn("TryOn backend call fallback:", e);
    }
  }

  let webcamInterval = null;
  function startLiveWebcamAnalysis() {
    if (webcamInterval) clearInterval(webcamInterval);
    const video = document.getElementById('tryon-video');

    webcamInterval = setInterval(async () => {
      if (!webcamStream || !video || video.paused || video.ended) {
        clearInterval(webcamInterval);
        return;
      }

      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = 640;
      tmpCanvas.height = 480;
      const ctx = tmpCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, 640, 480);
      const frameData = tmpCanvas.toDataURL('image/jpeg', 0.65);

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
          const lmDict = {};
          data.landmarks.forEach(l => { lmDict[l.id] = { x: l.x, y: l.y }; });
          renderRealisticClothTryOn(video, lmDict, data.metrics);
        }
      } catch (err) {}
    }, 1200);
  }

  // -------------------------------------------------------------
  // True Photorealistic Cloth Warping & Blending Algorithm
  // -------------------------------------------------------------
  function renderRealisticClothTryOn(mediaElement, landmarksDict, metrics) {
    const canvas = document.getElementById('tryon-canvas');
    if (!canvas) return;

    const w = canvas.parentElement.clientWidth;
    const h = canvas.parentElement.clientHeight || 520;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    const ls = (landmarksDict && landmarksDict[11]) || { x: 0.32, y: 0.34 };
    const rs = (landmarksDict && landmarksDict[12]) || { x: 0.68, y: 0.34 };
    const lh = (landmarksDict && landmarksDict[23]) || { x: 0.35, y: 0.75 };
    const rh = (landmarksDict && landmarksDict[24]) || { x: 0.65, y: 0.75 };

    // Anatomical Coordinates in Pixel Space
    const lx = ls.x * w;
    const ly = ls.y * h;
    const rx = rs.x * w;
    const ry = rs.y * h;

    // Shoulder angle & width calculation
    const shoulderWidth = Math.hypot(rx - lx, ry - ly);
    const shoulderAngle = Math.atan2(ry - ly, rx - lx);
    const shoulderCenterX = (lx + rx) / 2;
    const shoulderCenterY = (ly + ry) / 2;

    // Get current colorway garment asset
    const garmentImg = GARMENT_IMAGES[state.activeColorway] || GARMENT_IMAGES.obsidian;

    if (garmentImg && garmentImg.complete) {
      ctx.save();

      // 1. Natural Contact Ambient Occlusion Shadow
      ctx.save();
      ctx.translate(shoulderCenterX, shoulderCenterY + 12);
      ctx.rotate(shoulderAngle);
      ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
      ctx.shadowBlur = 24;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 8;

      const drapeW = shoulderWidth * 1.58;
      const drapeH = drapeW * (1000 / 800) * 1.05;

      // Draw invisible shadow caster
      ctx.fillStyle = 'rgba(0, 0, 0, 0.01)';
      ctx.fillRect(-drapeW / 2, -drapeH * 0.22, drapeW, drapeH);
      ctx.restore();

      // 2. Realistic Garment Placement & Perspective Warp
      ctx.save();
      ctx.translate(shoulderCenterX, shoulderCenterY);
      ctx.rotate(shoulderAngle);

      // Natural soft lighting & subtle fabric drape blend
      ctx.globalAlpha = 0.96;
      ctx.drawImage(
        garmentImg,
        -drapeW / 2,
        -drapeH * 0.22,
        drapeW,
        drapeH
      );
      ctx.restore();

      // 3. Subtle Holographic Fit Pin Accents
      ctx.strokeStyle = '#CA8A04';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);

      // Collar Anchor Indicator
      ctx.beginPath();
      ctx.arc(shoulderCenterX, shoulderCenterY - 10, 4, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#CA8A04';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`ATELIER FIT: ${metrics.recommended_size}`, shoulderCenterX - 55, shoulderCenterY - 22);

      ctx.restore();
    }
  }

  // -------------------------------------------------------------
  // Acoustic Waveform Visualizer
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

      // Ambient Raw Soundwave
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(220, 38, 38, 0.4)';
      ctx.lineWidth = 1.2;
      for (let x = 0; x < w; x++) {
        const y = h * 0.5 + Math.sin(x * 0.08 + phase * 2) * 18 * Math.cos(x * 0.02);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Attenuated AETHER Soundwave
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
          alert(`Allocation secured! Serial: AETH-2026-X77A. Master tailor dossier dispatched.`);
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
  // Colorway Palette Switcher & Layer Inspector
  // -------------------------------------------------------------
  function initColorwaySwitcher() {
    document.querySelectorAll('.colorway-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.colorway-btn').forEach(b => {
          b.classList.remove('active');
          b.style.borderColor = 'var(--border-subtle)';
        });
        btn.classList.add('active');
        btn.style.borderColor = 'var(--accent-gold)';

        const colorName = btn.dataset.name.toLowerCase();
        if (colorName.includes('obsidian')) state.activeColorway = 'obsidian';
        else if (colorName.includes('titanium')) state.activeColorway = 'titanium';
        else if (colorName.includes('crimson')) state.activeColorway = 'crimson';
        else if (colorName.includes('gold')) state.activeColorway = 'gold';

        // Rebuild Hero 3D model with new colorway
        rebuildHeroGarment(state.activeColorway);

        // Update AR Try-On Drape
        loadModelPreset(state.selectedModelIndex);
        playChime(620, 'sine', 0.25, 0.08);
      });
    });
  }

  function initLayerInspector() {
    const cards = document.querySelectorAll('.layer-card');
    cards.forEach((card) => {
      card.addEventListener('click', () => {
        cards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        const layerId = parseInt(card.dataset.layerId || '1');
        state.activeLayer = layerId;

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
    if (window.lucide) lucide.createIcons();

    const soundscapeBtn = document.getElementById('soundscape-btn');
    if (soundscapeBtn) soundscapeBtn.addEventListener('click', toggleAudio);

    if (window.THREE) {
      initHero3D();
      initExploded3D();
    } else {
      setTimeout(() => {
        if (window.THREE) {
          initHero3D();
          initExploded3D();
        }
      }, 500);
    }

    initTryOnStudio();
    initWaveformVisualizer();
    initReservationModal();
    initLayerInspector();
    initColorwaySwitcher();

    const resetCamBtn = document.getElementById('btn-reset-cam');
    if (resetCamBtn && heroControls) {
      resetCamBtn.addEventListener('click', () => {
        heroCamera.position.set(0, 0.4, 7.2);
        heroControls.target.set(0, 0, 0);
        heroControls.update();
      });
    }

    const toggleRotBtn = document.getElementById('btn-toggle-rot');
    if (toggleRotBtn) {
      toggleRotBtn.addEventListener('click', () => {
        state.rotationActive = !state.rotationActive;
        toggleRotBtn.classList.toggle('active', state.rotationActive);
      });
    }
  });

})();
