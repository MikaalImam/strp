let ws = null;
let isConnected = false;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const videoStream = document.getElementById('videoStream');
const status = document.getElementById('status');
const distanceValue = document.getElementById('distanceValue');

startBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);

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
        status.textContent = 'Streaming...';
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.error) {
            status.textContent = `Error: ${data.error}`;
            return;
        }
        
        // Display frame
        videoStream.src = `data:image/jpeg;base64,${data.frame}`;
        videoStream.classList.add('active');
        
        // Update angle
        if (data.angle !== null) {
            distanceValue.textContent = `${data.angle.toFixed(1)}°`;
        } else {
            distanceValue.textContent = '--';
        }


    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        status.textContent = 'Connection error';
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed');
        isConnected = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        videoStream.classList.remove('active');
        status.textContent = 'Connection closed';
        distanceValue.textContent = '--';
    };
}

function stopWebcam() {
    if (ws) {
        ws.close();
        ws = null;
    }
}
