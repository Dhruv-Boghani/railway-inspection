import React, { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Activity, CheckCircle, Circle, ArrowRight, Loader } from 'lucide-react';
import axios from 'axios';
import NoJobFound from '../components/NoJobFound';
import { useJob } from '../context/JobContext';

// SIDE Pipeline (LR/RL) - 7 stages
const SIDE_STAGES = [
    { id: '1', name: 'Wagon Detection & Counting' },
    { id: '2', name: 'Quality Assessment' },
    { id: '3', name: 'Image Enhancement' },
    { id: '4', name: 'Door Detection & Classification' },
    { id: '5', name: 'Damage Detection' },
    { id: '6', name: 'Wagon Number Extraction (OCR)' },
    { id: '7', name: 'Result Aggregation' },
];

// TOP Pipeline - 5 stages (no doors, no OCR)
const TOP_STAGES = [
    { id: '1', name: 'Wagon Detection & Counting' },
    { id: '2', name: 'Quality Assessment' },
    { id: '3', name: 'Image Enhancement' },
    { id: '4', name: 'Damage Detection' },
    { id: '5', name: 'Result Aggregation' },
];

// Progress thresholds for each pipeline
const SIDE_THRESHOLDS = [15, 30, 45, 60, 75, 90, 100]; // 7 stages
const TOP_THRESHOLDS = [20, 40, 60, 80, 100]; // 5 stages

const ProcessingPage = () => {
    const { jobId } = useParams();
    const navigate = useNavigate();
    const { setIsProcessing } = useJob();
    const [job, setJob] = useState(null);
    const [error, setError] = useState('');
    const [notFound, setNotFound] = useState(false);
    const [resultMissing, setResultMissing] = useState(false);
    // Track if job was already completed when we first loaded the page
    const wasAlreadyCompleted = useRef(null);
    const hasRedirected = useRef(false);

    // Determine pipeline type and stages based on camera_angle
    const isTopPipeline = job?.camera_angle === 'TOP';
    const STAGES = isTopPipeline ? TOP_STAGES : SIDE_STAGES;
    const THRESHOLDS = isTopPipeline ? TOP_THRESHOLDS : SIDE_THRESHOLDS;

    // Polling logic
    useEffect(() => {
        if (!jobId) {
            setNotFound(true);
            return;
        }

        let interval = null;

        const fetchStatus = async () => {
            try {
                const response = await axios.get(`http://localhost:8000/jobs/${jobId}`);
                const data = response.data;
                setJob(data);

                // On first fetch, check if job was already completed
                if (wasAlreadyCompleted.current === null) {
                    wasAlreadyCompleted.current = data.status === 'completed';

                    // Set processing state for sidebar animation
                    if (data.status === 'processing') {
                        setIsProcessing(true);
                    }

                    // If job was already completed when we loaded the page,
                    // verify that the result file actually exists
                    if (wasAlreadyCompleted.current) {
                        try {
                            await axios.get(`http://localhost:8000/jobs/${jobId}/result`);
                        } catch (resultErr) {
                            // Result file is missing - show upload page
                            if (resultErr.response && (resultErr.response.status === 404 || resultErr.response.status === 400)) {
                                setResultMissing(true);
                                if (interval) clearInterval(interval);
                                return;
                            }
                        }
                    }
                }

                // Only auto-redirect if:
                // 1. Job just finished (wasn't completed when we first loaded)
                // 2. We haven't already redirected
                if (data.status === 'completed' && !wasAlreadyCompleted.current && !hasRedirected.current) {
                    hasRedirected.current = true;
                    // Wait a moment so user sees 100% then redirect
                    setTimeout(() => {
                        navigate(`/results/${jobId}`);
                    }, 1000);
                } else if (data.status === 'failed') {
                    setError(data.error_message || 'Processing failed');
                }

                // Stop polling if job is completed or failed
                if (data.status === 'completed' || data.status === 'failed') {
                    setIsProcessing(false);
                    if (interval) {
                        clearInterval(interval);
                        interval = null;
                    }
                }
            } catch (err) {
                console.error('Polling error:', err);
                if (err.response && err.response.status === 404) {
                    setNotFound(true);
                    if (interval) clearInterval(interval);
                } else {
                    setError('Failed to connect to backend');
                }
            }
        };

        // Initial fetch
        fetchStatus();

        // Poll every 2 seconds only if job is still processing
        interval = setInterval(fetchStatus, 2000);
        return () => {
            if (interval) clearInterval(interval);
            setIsProcessing(false); // Reset on unmount
        };
    }, [jobId, navigate, setIsProcessing]);

    // Custom logic matching the backend progress mapping
    const isStageCompleted = (index) => {
        if (!job) return false;
        if (job.status === 'completed') return true;
        const progress = job.progress || 0;
        return progress >= THRESHOLDS[index];
    };

    if (notFound) {
        return <NoJobFound message="Job Not Found" />;
    }

    // Result file is missing (outputs folder was deleted)
    if (resultMissing) {
        return <NoJobFound message="Result Data Missing" />;
    }

    // Check for stale job - job completed but no output_dir or job is old
    // This handles the case when job exists in DB but output folder was deleted
    const isStaleJob = job && job.status === 'completed' && !job.output_dir;

    if (isStaleJob) {
        return <NoJobFound message="Job Data Expired" />;
    }

    if (error) {
        return (
            <div className="bg-red-50 p-8 rounded-xl border border-red-200 text-center">
                <h2 className="text-2xl font-bold text-red-700 mb-2">Processing Failed</h2>
                <p className="text-red-600">{error}</p>
                <button
                    onClick={() => navigate('/upload')}
                    className="mt-6 px-6 py-2 bg-white border border-red-200 text-red-600 rounded-lg hover:bg-red-50 font-medium"
                >
                    Try Again
                </button>
            </div>
        );
    }

    // Loading state - show while fetching job data
    if (!job) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh]">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                <p className="text-gray-500">Loading job status...</p>
            </div>
        );
    }

    return (
        <div className="space-y-8">
            {/* Header Card */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-4">
                        {/* Icon */}
                        <div className="bg-primary text-white p-2 rounded-lg">
                            <Activity className="w-10 h-10 animate-pulse" />
                        </div>

                        {/* Text */}
                        <div>
                            <h2 className="text-2xl font-bold text-primary">
                                Processing Inspection Video
                            </h2>
                            <p className="text-gray-500 text-md mt-1">
                                AI is analyzing your video for wagon defects and anomalies
                            </p>
                        </div>
                    </div>

                    {/* Pipeline Type Badge */}
                    {job && (
                        <div className="text-right">
                            <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${isTopPipeline
                                ? 'bg-purple-100 text-purple-700'
                                : 'bg-blue-100 text-blue-700'
                                }`}>
                                {isTopPipeline ? 'TOP Pipeline' : 'SIDE Pipeline'}
                            </span>
                            <p className="text-md text-gray-400 mt-1 pr-6">{STAGES.length} stages</p>
                        </div>
                    )}
                </div>
            </div>


            {/* Progress Bar */}
            <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
                <div className="flex justify-between items-end mb-2">
                    <span className="font-semibold text-gray-700">Overall Progress</span>
                    <span className="text-3xl font-bold text-primary">{Math.round(job?.progress || 0)}%</span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-4 overflow-hidden">
                    <div
                        className="bg-gradient-to-r from-accent-light to-accent-hover h-full rounded-full transition-all duration-500 ease-out"
                        style={{ width: `${job?.progress || 0}%` }}
                    />
                </div>
                <p className="text-sm text-gray-400 mt-2 text-right">
                    Current Step: <span className="text-primary font-medium">{job?.current_stage}</span>
                </p>
            </div>

            {/* Stages List */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="p-6 border-b border-gray-100 bg-gray-50/50">
                    <h3 className="font-semibold text-gray-800">Pipeline Stages</h3>
                </div>
                <div className="divide-y divide-gray-100">
                    {STAGES.map((stage, index) => {
                        const completed = isStageCompleted(index);
                        // Approximate active check: if not completed but previous is completed (or index 0)
                        const active = !completed && (index === 0 || isStageCompleted(index - 1));

                        // Get stage-specific count based on stage name
                        // Uses live_counts from backend for real-time updates
                        const getStageCount = () => {
                            if (!job) return null;

                            const stageName = stage.name.toLowerCase();
                            const liveCounts = job.live_counts || {};

                            // Stage 1: Wagon Detection & Counting
                            if (stageName.includes('wagon detection')) {
                                const count = liveCounts.wagons_counted || job.total_wagons || 0;
                                if (count > 0 || completed) {
                                    return { label: 'Wagons Counted', value: count, color: 'text-primary' };
                                }
                            }

                            // Stage 2: Quality Assessment
                            if (stageName.includes('quality')) {
                                const count = liveCounts.frames_assessed || (completed ? job.total_wagons : 0);
                                if (count > 0) {
                                    return { label: 'Frames Assessed', value: count, color: 'text-primary' };
                                }
                            }

                            // Stage 3: Image Enhancement
                            if (stageName.includes('enhancement')) {
                                const count = liveCounts.frames_enhanced || (completed ? job.total_wagons : 0);
                                if (count > 0) {
                                    return { label: 'Frames Enhanced', value: count, color: 'text-primary' };
                                }
                            }

                            // Stage 4: Door Detection (SIDE only)
                            if (stageName.includes('door')) {
                                const count = liveCounts.doors_detected || (completed ? (job.total_wagons * 2) : 0);
                                if (count > 0 || completed) {
                                    return { label: 'Doors Detected', value: count || (job.total_wagons * 2), color: 'text-primary' };
                                }
                            }

                            // Stage 4/5: Damage Detection (stage 4 for TOP, stage 5 for SIDE)
                            if (stageName.includes('damage')) {
                                const defects = liveCounts.defects_found !== undefined ? liveCounts.defects_found : job.defects_count;
                                // Strictly check active or completed to prevent showing "0" prematurely
                                if ((active || completed) && defects !== undefined) {
                                    const count = defects || 0;
                                    return { label: 'Defects Found', value: count, color: count > 0 ? 'text-primary' : 'text-primary' };
                                }
                            }

                            // Stage 6: OCR (SIDE only)
                            if (stageName.includes('ocr') || stageName.includes('number extraction')) {
                                // During processing, show "Crops Processed" (incrementing 1..43)
                                // After completion, show "Numbers Extracted" (successful wagons)
                                const cropCount = liveCounts.ocr_crops_processed;
                                const finalCount = liveCounts.numbers_extracted;

                                if (active && cropCount !== undefined) {
                                    return { label: 'Crops Processed', value: cropCount, color: 'text-primary' };
                                }
                                if (completed || finalCount !== undefined) {
                                    return { label: 'Numbers Extracted', value: finalCount || 0, color: 'text-primary' };
                                }
                            }

                            // Stage 7: Aggregation
                            if (stageName.includes('aggregation') && completed) {
                                return { label: 'Report Ready', value: '✓', color: 'text-primary' };
                            }

                            return null;
                        };

                        const stageCount = getStageCount();

                        return (
                            <div
                                key={stage.id}
                                className={`p-5 flex items-center justify-between transition-colors ${active ? 'bg-blue-50/40 border-l-4 border-l-accent' : 'border-l-4 border-l-transparent'}`}
                            >
                                <div className="flex items-center space-x-4">
                                    <div className={`w-8 h-8 rounded-full flex items-center justify-center border-2 
                                ${completed ? 'bg-green-100 border-green-500 text-green-600' :
                                            active ? 'bg-accent/10 border-accent text-accent' : 'bg-gray-50 border-gray-200 text-gray-300'}`}>
                                        {completed ? <CheckCircle className="w-5 h-5" /> :
                                            active ? <Loader className="w-4 h-4 animate-spin" /> :
                                                <span className="text-sm font-medium">{stage.id}</span>}
                                    </div>
                                    <div>
                                        <h4 className={`font-medium ${completed || active ? 'text-gray-800' : 'text-gray-400'}`}>
                                            {stage.name}
                                        </h4>
                                        {active && (
                                            <span className="text-xs text-accent font-medium animate-pulse">Processing...</span>
                                        )}
                                        {completed && (
                                            <span className="text-xs text-green-600 font-medium">Completed</span>
                                        )}
                                    </div>
                                </div>

                                {/* Dynamic Counts for each stage */}
                                {stageCount && (
                                    <div className="bg-white px-4 py-2 rounded-lg border shadow-sm text-right">
                                        <span className="text-xs text-gray-500 block">{stageCount.label}</span>
                                        <div className={`text-xl font-bold leading-tight ${stageCount.color}`}>
                                            {stageCount.value}
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div >
    );
};

export default ProcessingPage;
