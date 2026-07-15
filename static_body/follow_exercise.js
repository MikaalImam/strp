let ws = null;

const exerciseSelect = document.getElementById('exerciseSelect');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const restartBtn = document.getElementById('restartBtn');
const backBtn = document.getElementById('backBtn');

const videoStream = document.getElementById('videoStream');
const status = document.getElementById('status');
const stateValue = document.getElementById('stateValue');
const stepValue = document.getElementById('stepValue');
const matchValue = document.getElementById('matchValue');

startBtn.addEventListener('click', startSession);
stopBtn.addEventListener('click', stopSession);
restartBtn.addEventListener('click', restartExercise);

backBtn.addEventListener('click', backToHome);

function backToHome() {
    if (ws) {
        ws.close();
        ws = null;
    }
}


async function loadExercises() {
    try {
        const response = await fetch('/api/pose-exercises');
        const data = await response.json();
        const exercises = data.exercises || [];

        exerciseSelect.innerHTML = '';
        if (exercises.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No recorded exercises found';
            exerciseSelect.appendChild(option);
            startBtn.disabled = true;
            status.textContent = 'Record an exercise first.';
            return;
        }

        exercises.forEach((name) => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            exerciseSelect.appendChild(option);
        });
    } catch (error) {
        status.textContent = 'Failed to load exercise list.';
    }
}

function startSession() {
    const selected = exerciseSelect.value;
    if (!selected) {
        status.textContent = 'Please select an exercise.';
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/pose/follow/${encodeURIComponent(selected)}`;
    ws = new WebSocket(wsUrl);
    status.textContent = 'Connecting...';

    ws.onopen = () => {
        startBtn.disabled = true;
        exerciseSelect.disabled = true;
        stopBtn.disabled = false;
        restartBtn.disabled = false;
        status.textContent = `Following: ${selected}`;
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

        stateValue.textContent = data.state || '--';
        const current = Number(data.current_step ?? 0);
        const total = Number(data.total_steps ?? 0);
        stepValue.textContent = total > 0 ? `${Math.min(current + 1, total)} / ${total}` : '--';
        matchValue.textContent = `${Math.round(Number(data.match_fraction ?? 0) * 100)}%`;
    };

    ws.onclose = () => {
        startBtn.disabled = false;
        exerciseSelect.disabled = false;
        stopBtn.disabled = true;
        restartBtn.disabled = true;
        status.textContent = 'Connection closed';
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

function restartExercise() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }
    ws.send(JSON.stringify({ action: 'restart' }));
}

loadExercises();
