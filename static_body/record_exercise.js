let ws = null;
let isRecording = false;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const recordBtn = document.getElementById('recordBtn');
const clearBtn = document.getElementById('clearBtn');
const saveBtn = document.getElementById('saveBtn');
const backBtn = document.getElementById('backBtn');
const exerciseName = document.getElementById('exerciseName');

const videoStream = document.getElementById('videoStream');
const status = document.getElementById('status');
const recordingState = document.getElementById('recordingState');
const keyframeCount = document.getElementById('keyframeCount');
const elapsedTime = document.getElementById('elapsedTime');
const lastEvent = document.getElementById('lastEvent');

startBtn.addEventListener('click', startSession);
stopBtn.addEventListener('click', stopSession);
recordBtn.addEventListener('click', toggleRecording);
clearBtn.addEventListener('click', clearRecording);
saveBtn.addEventListener('click', saveRecording);
backBtn.addEventListener('click', backToHome);

function backToHome() {
    if (ws) {
        ws.close();
        ws = null;
    }
}


function setConnectedUI(connected) {
    startBtn.disabled = connected;
    stopBtn.disabled = !connected;
    recordBtn.disabled = !connected;
    clearBtn.disabled = !connected;
    saveBtn.disabled = !connected;
}

function startSession() {
    status.textContent = 'Connecting...';
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/pose/record`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        setConnectedUI(true);
        status.textContent = 'Connected';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
            status.textContent = `Error: ${data.error}`;
            return;
        }

        if (data.frame) {
            videoStream.src = `data:image/jpeg;base64,${data.frame}`;
            videoStream.classList.add('active');
        }

        isRecording = Boolean(data.recording);
        recordingState.textContent = isRecording ? 'Recording' : 'Idle';
        recordBtn.textContent = isRecording ? 'Stop Recording' : 'Start Recording';
        keyframeCount.textContent = String(data.num_keyframes ?? 0);
        elapsedTime.textContent = `${Number(data.elapsed ?? 0).toFixed(1)}s`;

        if (data.event) {
            lastEvent.textContent = data.event;
        }
    };

    ws.onclose = () => {
        setConnectedUI(false);
        isRecording = false;
        recordBtn.textContent = 'Start Recording';
        recordingState.textContent = 'Idle';
        status.textContent = 'Connection closed';
        videoStream.classList.remove('active');
    };

    ws.onerror = () => {
        status.textContent = 'Connection error';
    };
}

function stopSession() {
    if (ws) {
        ws.close();
        ws = null;
    }
}

function sendAction(action, extra = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }
    ws.send(JSON.stringify({ action, ...extra }));
}

function toggleRecording() {
    sendAction(isRecording ? 'stop_recording' : 'start_recording');
}

function clearRecording() {
    sendAction('clear_recording');
}

function saveRecording() {
    sendAction('save_recording', { name: exerciseName.value });
}
