const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const placeholder = document.getElementById("placeholder");
const dot = document.getElementById("dot");
const cameraStatus = document.getElementById("cameraStatus");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const captureBtn = document.getElementById("captureBtn");
const cameraSelect = document.getElementById("cameraSelect");
const refreshCameraBtn = document.getElementById("refreshCameraBtn");

let stream = null;

function setCameraStatus(on, text) {
  dot.classList.toggle("on", on);
  cameraStatus.textContent = text;
}

function showError(message) {
  document.getElementById("statusText").textContent = message;
  alert(message);
}

async function loadCameras() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera API is not supported by this browser.");
    }

    const permissionStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    permissionStream.getTracks().forEach((track) => track.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((device) => device.kind === "videoinput");
    cameraSelect.innerHTML = "";

    if (!cameras.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No camera detected";
      cameraSelect.appendChild(option);
      return;
    }

    cameras.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      cameraSelect.appendChild(option);
    });

    const externalCamera = cameras.find((camera) => {
      const name = (camera.label || "").toLowerCase();
      return (
        name.includes("usb") ||
        name.includes("external") ||
        name.includes("webcam") ||
        name.includes("hd camera") ||
        name.includes("logitech") ||
        name.includes("droidcam")
      );
    });

    cameraSelect.value = (externalCamera || cameras[0]).deviceId;
  } catch (error) {
    cameraSelect.innerHTML = '<option value="">Camera permission required</option>';
    setCameraStatus(false, "Camera Permission Required");
  }
}

async function startCamera() {
  try {
    if (!cameraSelect.value) {
      alert("Please select a camera first.");
      return;
    }

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: cameraSelect.value },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
    placeholder.style.display = "none";
    startBtn.disabled = true;
    stopBtn.disabled = false;
    captureBtn.disabled = false;
    cameraSelect.disabled = true;
    refreshCameraBtn.disabled = true;
    setCameraStatus(true, "Camera Connected");
  } catch (error) {
    setCameraStatus(false, "Camera Error");
    alert("Camera could not be opened.\n\n" + error.message + "\n\nTry selecting another camera.");
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  stream = null;
  video.srcObject = null;
  placeholder.style.display = "block";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  captureBtn.disabled = true;
  cameraSelect.disabled = false;
  refreshCameraBtn.disabled = false;
  setCameraStatus(false, "Camera Off");
}

function renderResult(data) {
  const day = data.predicted_day || data.raw_predicted_day || "Uncertain";
  document.getElementById("predictedDay").textContent = data.success && data.predicted_day
    ? String(day).toUpperCase()
    : (data.low_confidence ? "LOW CONFIDENCE" : "—");

  document.getElementById("developmentStage").textContent =
    data.development_stage || (data.low_confidence ? "Not assigned" : "—");

  document.getElementById("confidence").textContent =
    typeof data.confidence === "number" ? `${data.confidence.toFixed(2)}%` : "—";

  document.getElementById("statusText").textContent =
    data.status || data.message || "No result";

  const when = [data.date_display, data.time_display].filter(Boolean).join(" — ");
  document.getElementById("dateTime").textContent = when || "—";

  const list = document.getElementById("topPredictions");
  list.innerHTML = "";
  (data.top_predictions || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.day}: ${Number(item.confidence).toFixed(2)}%`;
    list.appendChild(li);
  });

  const preview = document.getElementById("capturedPreview");
  if (data.image) {
    preview.src = data.image + "?t=" + Date.now();
    preview.hidden = false;
  }

  (data.warnings || []).forEach((warning) => {
    const li = document.createElement("li");
    li.textContent = warning;
    list.appendChild(li);
  });
}

async function captureAndAnalyze() {
  if (!stream) {
    alert("Please start the camera first.");
    return;
  }
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    alert("Camera video is not ready yet.");
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const capturedImage = canvas.toDataURL("image/jpeg", 0.92);

  const preview = document.getElementById("capturedPreview");
  preview.src = capturedImage;
  preview.hidden = false;

  captureBtn.disabled = true;
  captureBtn.textContent = "Analyzing...";
  document.getElementById("statusText").textContent = "Saving image and running YOLO...";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: capturedImage }),
    });
    const result = await response.json();
    renderResult(result);

    if (!result.success) {
      alert(result.message || "Analysis failed.");
    }
  } catch (error) {
    showError("Failed to reach the Flask backend. Make sure py app.py is running.\n\n" + error.message);
  } finally {
    captureBtn.disabled = false;
    captureBtn.textContent = "CAPTURE & ANALYZE";
  }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
captureBtn.addEventListener("click", captureAndAnalyze);
refreshCameraBtn.addEventListener("click", loadCameras);
cameraSelect.addEventListener("change", async () => {
  if (stream) {
    stopCamera();
    await startCamera();
  }
});

loadCameras();
