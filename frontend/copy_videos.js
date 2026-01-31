const fs = require('fs');
const path = require('path');

const src1 = String.raw`e:\RUNNING project\railway-inspection\inputs\VID_20260201_013152.mp4`;
const dest1 = String.raw`e:\RUNNING project\railway-inspection\frontend\public\videos\test_video.mp4`;

const src2 = String.raw`e:\RUNNING project\railway-inspection\frontend\src\assets\VID_20260201_013152.mp4`;
const dest2 = String.raw`e:\RUNNING project\railway-inspection\frontend\public\videos\overview.mp4`;

try {
    console.log(`Copying ${src1} to ${dest1}...`);
    fs.copyFileSync(src1, dest1);
    console.log('Success 1');
} catch (e) {
    console.error('Error 1:', e);
}

try {
    console.log(`Copying ${src2} to ${dest2}...`);
    fs.copyFileSync(src2, dest2);
    console.log('Success 2');
} catch (e) {
    console.error('Error 2:', e);
}
