let ws = null;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const videoStream = document.getElementById('videoStream');
const status = document.getElementById('status');
const angleValue = document.getElementById('angleValue');
const videoName = document.getElementById('videoName');

startBtn.addEventListener('click', startExerciseStream);
stopBtn.addEventListener('click', stopExerciseStream);

function startExerciseStream() {
    status.textContent = 'Connecting...';

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/exercise`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        status.textContent = 'Streaming exercise video...';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
            status.textContent = `Error: ${data.error}`;
            stopExerciseStream();
            return;
        }

        videoStream.src = `data:image/jpeg;base64,${data.frame}`;
        videoStream.classList.add('active');

        if (data.angle !== null) {
            angleValue.textContent = `${data.angle.toFixed(1)} deg`;
        } else {
            angleValue.textContent = '--';
        }

        videoName.textContent = data.video || '--';
    };

    ws.onerror = () => {
        status.textContent = 'Connection error';
    };

    ws.onclose = () => {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        status.textContent = 'Stream stopped';
        videoStream.classList.remove('active');
    };
}

function stopExerciseStream() {
    if (ws) {
        ws.close();
        ws = null;
    }
}
