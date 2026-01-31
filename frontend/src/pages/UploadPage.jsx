import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useJob } from '../context/JobContext';
import axios from 'axios';
import { Upload, Video, Camera, ArrowRight, Loader2, Cpu, FileCheck, CheckCircle } from 'lucide-react';

const UploadPage = () => {
    const [file, setFile] = useState(null);
    const [cameraAngle, setCameraAngle] = useState('LR');
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const fileInputRef = useRef(null);
    const navigate = useNavigate();
    const { setCurrentJobId } = useJob();

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreviewUrl(URL.createObjectURL(selectedFile));
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && (droppedFile.type.startsWith('video/') || droppedFile.type.startsWith('image/'))) {
            setFile(droppedFile);
            setPreviewUrl(URL.createObjectURL(droppedFile));
        }
    };

    const handleSubmit = async () => {
        if (!file) return;

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('direction', cameraAngle === 'TOP' ? 'LR' : cameraAngle);
        formData.append('camera_angle', cameraAngle);

        try {
            const backendUrl = import.meta.env.VITE_BACKEND_URL;
            const response = await axios.post(`${backendUrl}/jobs/`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            const jobId = response.data.id;
            setCurrentJobId(jobId);
            navigate(`/processing/${jobId}`);
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Upload failed. Please try again.');
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="min-h-[calc(100vh-6rem)] flex flex-col gap-6">
            {/* Header */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-start space-x-4">
                    <div className="bg-primary text-white p-3 rounded-lg">
                        <Upload className="w-8 h-8" />
                    </div>
                    <div className="flex-1">
                        <h2 className="text-2xl font-bold text-primary">Upload Inspection Media</h2>
                        <p className="text-gray-500 mt-1">
                            Upload wagon inspection videos for automated AI-powered analysis
                        </p>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Upload Area - Takes 2 columns */}
                <div className="lg:col-span-2 flex">
                    <div
                        className={`bg-white rounded-xl shadow-sm border-2 border-dashed w-full min-h-[460px] flex flex-col items-center justify-center p-8 cursor-pointer
                            ${file ? 'border-accent/50 bg-accent/5' : 'border-gray-300 hover:border-primary/30 hover:bg-gray-50'}`}
                        onClick={() => !file && fileInputRef.current?.click()}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            className="hidden"
                            accept="video/*,image/*"
                            onChange={handleFileChange}
                        />

                        {file ? (
                            <div className="w-full h-full flex flex-col items-center justify-center">
                                {file.type.startsWith('video/') ? (
                                    <video
                                        src={previewUrl}
                                        className="max-h-[350px] w-auto rounded-lg shadow-md"
                                        controls
                                        muted
                                    />
                                ) : (
                                    <img
                                        src={previewUrl}
                                        alt="Preview"
                                        className="max-h-[350px] w-auto rounded-lg shadow-md object-contain"
                                    />
                                )}
                                <div className="mt-4 flex items-center justify-center gap-4">
                                    <span className="text-gray-700 font-medium">{file.name}</span>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setFile(null);
                                            setPreviewUrl(null);
                                            if (fileInputRef.current) fileInputRef.current.value = '';
                                        }}
                                        className="text-red-500 hover:text-red-600 text-sm font-medium px-3 py-1 rounded-lg hover:bg-red-50"
                                    >
                                        Remove
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="flex items-center gap-4 mb-6">
                                    <div className="p-4 bg-gray-100 rounded-xl">
                                        <Video className="w-7 h-7 text-gray-500" />
                                    </div>
                                    <div className="p-4 bg-gray-100 rounded-xl">
                                        <Camera className="w-7 h-7 text-gray-500" />
                                    </div>
                                </div>
                                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                                    Drag & Drop Media Here
                                </h3>
                                <p className="text-gray-400 mb-6">or click to browse your files</p>
                                <div className="flex items-center gap-3">
                                    <span className="px-4 py-1.5 bg-gray-100 text-gray-600 text-sm rounded-full font-medium">MP4</span>
                                    <span className="px-4 py-1.5 bg-gray-100 text-gray-600 text-sm rounded-full font-medium">AVI</span>
                                    <span className="px-4 py-1.5 bg-gray-100 text-gray-600 text-sm rounded-full font-medium">MKV</span>
                                </div>
                            </>
                        )}
                    </div>
                </div>

                {/* Settings Panel */}
                <div className="bg-white rounded-xl p-6 shadow-sm flex flex-col">
                    <h2 className="text-xl font-semibold text-gray-900 mb-6">Inspection Settings</h2>

                    <div className="mb-6">
                        <label className="block text-md font-medium text-gray-700 mb-3">Camera Angle</label>
                        <div className="grid grid-cols-3 gap-2">
                            <button
                                onClick={() => setCameraAngle('LR')}
                                className={`py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${cameraAngle === 'LR'
                                    ? 'bg-primary text-white shadow-md'
                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                            >
                                Left to Right
                            </button>
                            <button
                                onClick={() => setCameraAngle('RL')}
                                className={`py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${cameraAngle === 'RL'
                                    ? 'bg-primary text-white shadow-md'
                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                            >
                                Right to Left
                            </button>
                            <button
                                onClick={() => setCameraAngle('TOP')}
                                className={`py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${cameraAngle === 'TOP'
                                    ? 'bg-primary text-white shadow-md'
                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                            >
                                Top View
                            </button>
                        </div>
                        <p className="text-sm text-gray-400 mt-3">
                            {cameraAngle === 'TOP'
                                ? 'Top view: Full pipeline with floor damage detection from top angle (no door detection and OCR analysis).'
                                : 'Side view: Full pipeline with doors detection and classification, damage detection, and OCR detection.'}
                        </p>
                    </div>

                    <div className="bg-red-50 border border-red-500 rounded-lg p-4 mb-6">
                        <p className="text-sm text-red-700">
                            <span className="font-semibold"> Benchmark Note: </span> This application utilizes a multi-agent pipeline involving ~1.4GB of model weights. While local inference on an NVIDIA GPU achieves a runtime of 4-5 minutes, the current Hugging Face Space runs on 2 vCPUs (16GB RAM) without GPU acceleration. Consequently, users should expect a processing duration of 20-30 minutes for the full pipeline.
                        </p>
                    </div>



                    <div className="mt-auto">
                        <button
                            onClick={handleSubmit}
                            disabled={!file || isUploading}
                            className={`w-full py-3.5 rounded-lg font-semibold text-white flex items-center justify-center gap-2 transition-all ${!file || isUploading
                                ? 'bg-gray-300 cursor-not-allowed'
                                : 'bg-primary hover:bg-primary-light shadow-lg hover:shadow-xl'
                                }`}
                        >
                            {isUploading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    <span>Processing...</span>
                                </>
                            ) : (
                                <>
                                    <span>Start Video Inspection</span>
                                    <ArrowRight className="w-5 h-5" />
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-400 hover:shadow-lg transition-shadow">
                    <div className="flex items-start gap-4">
                        <div className="bg-primary/10 p-3 rounded-xl">
                            <Upload className="w-5 h-5 text-primary" />
                        </div>
                        <div className="flex-1">
                            <h3 className="font-semibold text-primary">1. Upload Media</h3>
                            <p className="text-xs text-accent mb-2">Universal Format Support</p>
                            <p className="text-sm text-gray-600 leading-relaxed">
                                Seamlessly upload full rake videos or spot-check photos. Our system automatically detects the format and adjusts the processing pipeline.
                            </p>
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-400 hover:shadow-lg transition-shadow">
                    <div className="flex items-start gap-4">
                        <div className="bg-primary/10 p-3 rounded-xl">
                            <Cpu className="w-5 h-5 text-primary" />
                        </div>
                        <div className="flex-1">
                            <h3 className="font-semibold text-primary">2. AI Processing</h3>
                            <p className="text-xs text-accent mb-2">Real-time Analysis</p>
                            <p className="text-sm text-gray-600 leading-relaxed">
                                Advanced computer vision algorithms scan every frame to detect structural damage, corrosion, and safety compliance issues instantly.
                            </p>
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-400 hover:shadow-lg transition-shadow">
                    <div className="flex items-start gap-4">
                        <div className="bg-primary/10 p-3 rounded-xl">
                            <CheckCircle className="w-5 h-5 text-primary" />
                        </div>
                        <div className="flex-1">
                            <h3 className="font-semibold text-primary">3. Instant Results</h3>
                            <p className="text-xs text-accent mb-2">Actionable Intelligence</p>
                            <p className="text-sm text-gray-600 leading-relaxed">
                                Receive detailed health reports, defect confidence scores, and automated alerts for critical maintenance requirements.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UploadPage;
