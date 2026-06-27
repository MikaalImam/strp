const videoStream = document.getElementById('videoStream');
const status = document.getElementById('status');

const angleValue = document.getElementById('angleValue');
const videoName = document.getElementById('videoName');

const VIDEO_FILE = 'shoulder_abduction_with_angles.mp4';
// const VIDEO_FILE = 'shoulder_abduction_with_angles.mp4';

const VIDEO_URL = `/exercises/${VIDEO_FILE}`;

videoStream.src = VIDEO_URL;
videoStream.controls = true;
videoStream.loop = true;
videoStream.preload = 'metadata';
videoStream.load();


videoStream.addEventListener('loadedmetadata', () => {
    videoName.textContent = VIDEO_FILE;
    angleValue.textContent = 'Drawn on video';
    status.style.display = 'none';

    console.log('Video duration:', videoStream.duration);
});

videoStream.addEventListener('play', () => {
    status.style.display = 'none';
});

videoStream.addEventListener('seeking', () => {
    status.style.display = 'none';
});

videoStream.addEventListener('seeked', () => {
    status.style.display = 'none';
});

videoStream.addEventListener('error', () => {
    status.style.display = 'block';
    status.textContent = `Could not load ${VIDEO_FILE}`;

    videoName.textContent = '--';
    angleValue.textContent = '--';

    console.log('Video error:', videoStream.error);
});