let ws = null;
let isConnected = false;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const videoStream = document.getElementById('videoStream');
const status = document.getElementById('status');
const angleValueRight = document.getElementById('angleValueRight');
const angleValueLeft = document.getElementById('angleValueLeft');
const leftRectangle = document.getElementById('leftRectangle');
const rightRectangle = document.getElementById('rightRectangle');

startBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);

function updateRectangleColor(rectangle, angle) {
    rectangle.classList.remove('too-low', 'correct', 'too-high');

    if (angle === null || angle === undefined) {
        return;
    }

    if (angle < 100) {
        rectangle.classList.add('too-low');
    } else if (angle >= 100 && angle <= 105) {
        rectangle.classList.add('correct');
    } else {
        rectangle.classList.add('too-high');
    }
}

function startWebcam() {
    status.textContent = 'Connecting...';
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        isConnected = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        status.style.display = 'none';
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        console.log(data);

        if (data.error) {
            status.textContent = `Error: ${data.error}`;
            return;
        }

        videoStream.src = `data:image/jpeg;base64,${data.frame}`;
        videoStream.classList.add('active');

        if (data.angle_right !== null && data.angle_right !== undefined) {
            angleValueRight.textContent = `${Number(data.angle_right).toFixed(1)}°`;
            updateRectangleColor(rightRectangle, data.angle_right);
        } else {
            angleValueRight.textContent = '--';
        }

        if (data.angle_left !== null && data.angle_left !== undefined) {
            angleValueLeft.textContent = `${Number(data.angle_left).toFixed(1)}°`;
            updateRectangleColor(leftRectangle, data.angle_left);
        } else {
            angleValueLeft.textContent = '--';
        }
    };    

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        status.style.display = 'block';
        status.textContent = 'Connection error';
    };
    
    ws.onclose = () => {
    console.log('WebSocket closed');
    isConnected = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    videoStream.classList.remove('active');

    status.style.display = 'block';
    status.textContent = 'Connection closed';

    angleValueRight.textContent = '--';
    angleValueLeft.textContent = '--';
    updateRectangleColor(rightRectangle, null);
    updateRectangleColor(leftRectangle, null);
};
}

function stopWebcam() {
    if (ws) {
        ws.close();
        ws = null;
    }
}
