import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { ChevronRight, Eye, AlertCircle, Type, Box, CheckCircle, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import { useJob } from '../context/JobContext';
import NoJobFound from '../components/NoJobFound';


// Wagon Number Parser Utility
const parseWagonNumber = (number) => {
    if (!number || typeof number !== 'string' || number.length !== 11 || !/^\d{11}$/.test(number)) {
        return null;
    }

    try {
        const digits = number.split('').map(Number);
        if (digits.some(isNaN)) return null;

        // Calculate check digit
        const evenPos = [1, 3, 5, 7, 9]; // Indices for positions 2,4,6,8,10
        const oddPos = [0, 2, 4, 6, 8];  // Indices for positions 1,3,5,7,9

        const s1 = evenPos.reduce((sum, i) => sum + digits[i], 0) * 3;
        const s2 = oddPos.reduce((sum, i) => sum + digits[i], 0);
        const s3 = s1 + s2;
        const nextMult10 = Math.ceil(s3 / 10) * 10;
        const expectedCheck = nextMult10 - s3;
        const actualCheck = digits[10];

        const isValid = expectedCheck === actualCheck;

        // Decode components
        const wagonType = number.substring(0, 2);
        const owningRailway = number.substring(2, 4);
        const yearDigits = number.substring(4, 6);
        const serial = number.substring(6, 10);

        // Determine full year
        const yearInt = parseInt(yearDigits);
        const fullYear = yearInt <= 30 ? `20${yearDigits}` : `19${yearDigits}`;

        return {
            fullNumber: number,
            isValid,
            expectedCheck,
            actualCheck,
            wagonType,
            owningRailway,
            year: fullYear,
            yearDigits,
            serial
        };
    } catch (e) {
        console.error("Error parsing wagon number", e);
        return null;
    }
};

// Wagon Type Code Lookup (c1,c2)
const getWagonType = (code) => {
    const numCode = parseInt(code);
    if (numCode >= 10 && numCode <= 29) return 'Open Wagon';
    if (numCode >= 30 && numCode <= 39) return 'Covered Wagon';
    if (numCode >= 40 && numCode <= 54) return 'Tank Wagon';
    if (numCode >= 55 && numCode <= 69) return 'Flat Wagon';
    if (numCode >= 70 && numCode <= 79) return 'Hopper Wagon';
    if (numCode >= 80 && numCode <= 84) return 'Well Wagon';
    if (numCode >= 85 && numCode <= 89) return 'Brake Van';
    return 'Unknown Type';
};

// Owning Railway Code Lookup (c3,c4)
const railwayNames = {
    '01': 'Central Railway',
    '02': 'Eastern Railway',
    '03': 'Northern Railway',
    '04': 'North East Railway',
    '05': 'Northeast Frontier Railway',
    '06': 'Southern Railway',
    '07': 'South Eastern Railway',
    '08': 'Western Railway',
    '09': 'South Central Railway',
    '10': 'East Central Railway',
    '11': 'North Western Railway',
    '12': 'East Coast Railway',
    '13': 'North Central Railway',
    '14': 'South East Central Railway',
    '15': 'South Western Railway',
    '16': 'West Central Railway',
    '25': 'CONCOR',
    '26': 'Private Party'
};

const getRailwayName = (code) => railwayNames[code] || 'Unknown Railway';

const AnalysisPage = () => {
    const { jobId } = useParams();
    const { setCurrentJobId } = useJob();
    const [job, setJob] = useState(null);
    const [results, setResults] = useState(null);
    const [selectedWagonId, setSelectedWagonId] = useState(null);
    const [viewMode, setViewMode] = useState('doors'); // 'doors', 'damage', 'combined', 'ocr'
    const [zoomLevel, setZoomLevel] = useState(1);
    const [isDragging, setIsDragging] = useState(false);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [notFound, setNotFound] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!jobId) {
            setNotFound(true);
            return;
        }
        if (jobId) setCurrentJobId(jobId);

        const fetchData = async () => {
            const backendUrl = import.meta.env.VITE_BACKEND_URL;
            try {
                const jobRes = await axios.get(`${backendUrl}/jobs/${jobId}`);
                setJob(jobRes.data);
                const resultRes = await axios.get(`${backendUrl}/jobs/${jobId}/result`);
                setResults(resultRes.data);

                // Handle both TOP and SIDE pipeline data structures
                const pipelineType = resultRes.data.pipeline_type || 'SIDE';
                const damageData = pipelineType === 'TOP'
                    ? (resultRes.data.stage4_damage || {})
                    : (resultRes.data.stage5_damage || resultRes.data.summary?.stage5_damage || {});

                // Enhanced Sorting Logic
                const sortedKeys = Object.keys(damageData).sort((a, b) => {
                    const numA = parseInt(a.match(/\d+/)?.[0] || "0");
                    const numB = parseInt(b.match(/\d+/)?.[0] || "0");
                    return numA - numB;
                });

                if (sortedKeys.length > 0) setSelectedWagonId(sortedKeys[0]);

                if (pipelineType === 'TOP') {
                    setViewMode('damage');
                }

            } catch (err) {
                console.error("Error fetching analysis data", err);
                if (err.response && err.response.status === 404) {
                    setNotFound(true);
                } else {
                    setError(err.message || "Failed to load data");
                }
            }
        };
        fetchData();
    }, [jobId]);

    // Reset zoom
    useEffect(() => {
        setZoomLevel(1);
        setPosition({ x: 0, y: 0 });
    }, [selectedWagonId, viewMode]);

    // Set default viewMode
    useEffect(() => {
        if (job?.camera_angle === 'TOP' && (viewMode === 'doors' || viewMode === 'ocr')) {
            setViewMode('damage');
        }
    }, [job]);

    if (notFound) return <NoJobFound />;
    if (error) return <div className="p-10 text-red-600">Error: {error}</div>;

    if (!job || !results || !selectedWagonId) {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-50">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                    <p className="text-gray-600 font-medium animate-pulse">Loading analysis...</p>
                    {job && <p className="text-xs text-gray-400 mt-2">Initializing...</p>}
                </div>
            </div>
        );
    }

    // Zoom handlers
    const handleZoomIn = () => setZoomLevel(prev => Math.min(prev + 0.25, 3));
    const handleZoomOut = () => setZoomLevel(prev => Math.max(prev - 0.25, 0.5));
    const handleResetZoom = () => {
        setZoomLevel(1);
        setPosition({ x: 0, y: 0 });
    };

    const handleMouseDown = (e) => {
        if (zoomLevel > 1) {
            setIsDragging(true);
            setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
        }
    };

    const handleMouseMove = (e) => {
        if (isDragging && zoomLevel > 1) {
            setPosition({
                x: e.clientX - dragStart.x,
                y: e.clientY - dragStart.y
            });
        }
    };

    const handleMouseUp = () => setIsDragging(false);
    const handleMouseLeave = () => setIsDragging(false);

    const handleWheel = (e) => {
        // e.preventDefault(); // removed to prevent passive event listener issues
        if (e.deltaY < 0) handleZoomIn();
        else handleZoomOut();
    };

    // Safe Data Extraction
    const pipelineType = results.pipeline_type || 'SIDE';
    const damageData = pipelineType === 'TOP'
        ? (results.stage4_damage || {})
        : (results.stage5_damage || results.summary?.stage5_damage || {});

    const cameraAngle = pipelineType === 'TOP' ? 'TOP' : (job.camera_angle || 'LR');

    const doorData = results.stage4_doors || results.summary?.stage4_doors || {};
    const ocrData = results.stage6_ocr?.results || results.stage6_ocr || results.summary?.stage6_ocr?.results || results.summary?.stage6_ocr || {};

    const currentWagonStats = damageData[selectedWagonId] || {};
    const currentWagonDoors = doorData[selectedWagonId] || {};

    // OCR Handling (Array vs Object safety)
    let currentWagonOCR = {};
    if (Array.isArray(ocrData)) {
        currentWagonOCR = ocrData.find(w => w.wagon_name === selectedWagonId || w.wagon_name === `${selectedWagonId}_enhanced`) || {};
    } else if (typeof ocrData === 'object') {
        currentWagonOCR = ocrData[selectedWagonId] || ocrData[`${selectedWagonId}_enhanced`] || {};
    }

    // Image URL Helper
    const getImageUrl = (type) => {
        try {
            const backendUrl = import.meta.env.VITE_BACKEND_URL;
            let absolutePath = "";
            if (type === 'original' || type === 'enhanced' || type === 'damage') {
                if (currentWagonStats.annotated_image_path) absolutePath = currentWagonStats.annotated_image_path;
            }
            if (type === 'doors') {
                if (currentWagonDoors.annotated_image_path) absolutePath = currentWagonDoors.annotated_image_path;
            }
            if (type === 'ocr') {
                if (currentWagonOCR.annotated_image_path) {
                    absolutePath = currentWagonOCR.annotated_image_path;
                } else {
                    // Use job's output_dir for job-specific path
                    const baseName = currentWagonOCR.wagon_name || `${selectedWagonId}_enhanced`;
                    const craftImageName = `${baseName}_stage6_craft_annotated.jpg`;
                    // Get relative path from job's output_dir
                    if (job?.output_dir) {
                        const outputDirParts = job.output_dir.split('outputs');
                        if (outputDirParts.length > 1) {
                            const jobFolder = outputDirParts[1].replace(/\\/g, '/');
                            return `${backendUrl}/outputs${jobFolder}/stage6_craft_annotated/${craftImageName}`;
                        }
                    }
                    // Fallback to old path format
                    return `${backendUrl}/outputs/stage6_craft_annotated/${craftImageName}`;
                }
            }

            if (!absolutePath) return "https://placehold.co/800x400/e2e8f0/1e293b?text=Image+Not+Available";

            const parts = absolutePath.split('outputs');
            if (parts.length > 1) {
                const relative = parts[1].replace(/\\/g, '/');
                return `${backendUrl}/outputs${relative}`;
            }
            return "https://placehold.co/800x400/e2e8f0/1e293b?text=Invalid+Path";
        } catch (e) {
            console.error("Error generating image URL", e);
            return "https://placehold.co/800x400/aa0000/ffffff?text=Error";
        }
    };

    const displayImage = getImageUrl(viewMode);

    const sortedWagonKeys = Object.keys(damageData).sort((a, b) => {
        const numA = parseInt(a.match(/\d+/)?.[0] || "0");
        const numB = parseInt(b.match(/\d+/)?.[0] || "0");
        return numA - numB;
    });

    // Render Safe
    return (
        <div className="flex h-screen overflow-hidden">
            {/* Sidebar List */}
            <div className="w-80 glass-panel border-r-0 rounded-r-2xl flex flex-col h-full z-10 my-4 ml-20 backdrop-blur-2xl">
                <div className="p-4 border-b border-indigo-100/50 font-bold text-slate-700 text-base flex-shrink-0 flex justify-between items-center bg-white/40 backdrop-blur-sm rounded-tr-2xl">
                    <span>Wagons</span>
                    <span className="bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full text-xs font-bold border border-indigo-200">{sortedWagonKeys.length}</span>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {sortedWagonKeys.map(id => {
                        const stats = damageData[id] || {};
                        const issues = (stats.total_detections || 0);
                        return (
                            <button
                                key={id}
                                onClick={() => setSelectedWagonId(id)}
                                className={`w-full p-4 flex items-center justify-between border-b border-indigo-50/50 hover:bg-white/60 transition-all duration-200 text-left group
                                    ${selectedWagonId === id ? 'bg-white/80 border-l-4 border-l-primary shadow-sm' : 'border-l-4 border-l-transparent'}`}
                            >
                                <div>
                                    <span className={`font-medium block capitalize text-base transition-colors ${selectedWagonId === id ? 'text-primary' : 'text-slate-600 group-hover:text-slate-800'}`}>{id.replace('_', ' ')}</span>
                                    <span className="text-sm text-slate-400">
                                        {issues === 0 ? 'Good Condition' : `${issues} Issues`}
                                    </span>
                                </div>
                                {issues > 0 ? (
                                    <AlertCircle className="w-4 h-4 text-danger" />
                                ) : (
                                    <div className="w-2.5 h-2.5 rounded-full bg-success mr-1 shadow-[0_0_10px_rgba(34,197,94,0.4)]"></div>
                                )}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col h-full overflow-hidden p-4 gap-4">
                <div className="glass-panel rounded-2xl flex flex-col h-full overflow-hidden">
                    {/* Toolbar */}
                    <div className="p-4 border-b border-indigo-100/50 flex justify-between items-center bg-white/40 backdrop-blur-sm flex-shrink-0 z-20">
                        <h2 className="font-bold text-xl text-slate-800 capitalize flex items-center gap-2">
                             <span className="text-primary">{selectedWagonId?.replace('_', ' ')}</span> Analysis
                        </h2>
                        <div className="flex bg-slate-100/50 rounded-xl p-1 border border-white/40 shadow-inner backdrop-blur-sm">
                            {cameraAngle !== 'TOP' && (
                                <button
                                    onClick={() => setViewMode('doors')}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center space-x-2 transition-all
                                    ${viewMode === 'doors' ? 'bg-white text-primary shadow-md shadow-indigo-500/10' : 'text-slate-500 hover:bg-white/50 hover:text-slate-700'}`}
                                >
                                    <Box className="w-4 h-4" />
                                    <span>Doors</span>
                                </button>
                            )}
                            <button
                                onClick={() => setViewMode('damage')}
                                className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center space-x-2 transition-all
                                    ${viewMode === 'damage' ? 'bg-white text-primary shadow-md shadow-indigo-500/10' : 'text-slate-500 hover:bg-white/50 hover:text-slate-700'}`}
                            >
                                <AlertCircle className="w-4 h-4" />
                                <span>Damage</span>
                            </button>
                            {cameraAngle !== 'TOP' && (
                                <button
                                    onClick={() => setViewMode('ocr')}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center space-x-2 transition-all
                                    ${viewMode === 'ocr' ? 'bg-white text-primary shadow-md shadow-indigo-500/10' : 'text-slate-500 hover:bg-white/50 hover:text-slate-700'}`}
                                >
                                    <Type className="w-4 h-4" />
                                    <span>OCR</span>
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Image Viewer */}
                    <div
                        className="flex-1 bg-slate-50/50 relative flex items-center justify-center p-4 min-h-0 overflow-hidden backdrop-blur-sm"
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseLeave}
                        onWheel={handleWheel}
                        style={{ cursor: isDragging ? 'grabbing' : (zoomLevel > 1 ? 'grab' : 'default') }}
                    >
                        <img
                            src={displayImage}
                            alt="Wagon Analysis"
                            className="max-h-full max-w-full object-contain rounded shadow-lg border border-white/20 select-none"
                            style={{
                                transform: `scale(${zoomLevel}) translate(${position.x / zoomLevel}px, ${position.y / zoomLevel}px)`,
                                transition: isDragging ? 'none' : 'transform 0.2s ease-out'
                            }}
                            onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = "https://placehold.co/800x400/e2e8f0/1e293b?text=Image+Load+Error";
                            }}
                            draggable={false}
                        />
                        {/* Zoom Controls Overlay */}
                        <div className="absolute bottom-4 right-4 flex flex-col space-y-2">
                            <button onClick={handleZoomIn} className="bg-white p-4 rounded-lg shadow-lg border border-gray-200"><ZoomIn className="w-4 h-4" /></button>
                            <button onClick={handleZoomOut} className="bg-white p-4 rounded-lg shadow-lg border border-gray-200"><ZoomOut className="w-4 h-4" /></button>
                            <button onClick={handleResetZoom} className="bg-white p-4 rounded-lg shadow-lg border border-gray-200"><RotateCcw className="w-4 h-4" /></button>
                            <div className="bg-white px-2 py-1 rounded-lg shadow-lg border border-gray-200 text-xs text-center">{Math.round(zoomLevel * 100)}%</div>
                        </div>
                    </div>

                    {/* Bottom Panel */}
                    <div className={`p-4 border-t border-indigo-100/50 grid gap-4 bg-white/30 backdrop-blur-sm flex-shrink-0 ${pipelineType === 'TOP' ? 'grid-cols-1' : 'grid-cols-1 md:grid-cols-3'}`} style={{ gridAutoRows: '1fr' }}>
                        {/* OCR Section */}
                        {pipelineType !== 'TOP' && (() => {
                            // Helper to validate number length (max 22 digits)
                            const isValidNumberLength = (num) => {
                                if (!num) return false;
                                const digitsOnly = num.replace(/\D/g, '');
                                return digitsOnly.length <= 22;
                            };

                            // Filter detected_number if > 22 digits
                            const validDetectedNumber = currentWagonOCR.detected_number && isValidNumberLength(currentWagonOCR.detected_number)
                                ? currentWagonOCR.detected_number
                                : null;

                            // Filter candidate_numbers to only those <= 22 digits
                            const validCandidates = (currentWagonOCR.candidate_numbers || []).filter(isValidNumberLength);

                            // Filter crop_texts to only those <= 22 digits
                            const validCropTexts = (currentWagonOCR.crop_texts || [])
                                .filter(t => t && t.toLowerCase() !== 'empty' && isValidNumberLength(t));
                            // Parse the number if valid to get details
                            const parsedWagon = validDetectedNumber ? parseWagonNumber(validDetectedNumber) : null;

                            return (
                                <div className="glass-card rounded-xl p-4 h-full flex flex-col">
                                    <div className="flex items-center space-x-2 mb-3">
                                        <div className="bg-primary/10 p-1.5 rounded-lg border border-primary/10">
                                            <Type className="w-4 h-4 text-primary" />
                                        </div>
                                        <h4 className="text-sm font-bold text-slate-700 uppercase tracking-wide">Wagon Number</h4>
                                    </div>
                                    {validDetectedNumber && parsedWagon ? (
                                        <div className="bg-success/5 p-4 rounded-lg border border-success/20 shadow-sm flex-1">
                                            {/* Header with Number */}
                                            <div className="text-center pb-3 mb-3 border-b border-success/20">
                                                <div className="flex items-center justify-center gap-2 mb-2">
                                                    <CheckCircle className="w-5 h-5 text-success" />
                                                    <span className="text-sm font-bold text-success uppercase tracking-wide">Verified Wagon</span>
                                                </div>
                                                <div className="font-mono text-2xl font-bold text-primary tracking-[0.2em]">{validDetectedNumber}</div>
                                            </div>
                                            {/* Wagon Details - 2x2 Grid */}
                                            <div className="grid grid-cols-2 gap-3">
                                                <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
                                                    <div className="text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Type</div>
                                                    <div className="text-sm font-bold text-gray-800 leading-tight">{getWagonType(parsedWagon.wagonType)}</div>
                                                    <div className="text-sm text-primary font-mono mt-1">code-{parsedWagon.wagonType}</div>
                                                </div>
                                                <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
                                                    <div className="text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Railway</div>
                                                    <div className="text-sm font-bold text-gray-800 leading-tight">{getRailwayName(parsedWagon.owningRailway)}</div>
                                                    <div className="text-sm text-primary font-mono mt-1">code-{parsedWagon.owningRailway}</div>
                                                </div>
                                                <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
                                                    <div className="text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Year Built</div>
                                                    <div className="text-lg font-bold text-gray-800">{parsedWagon.year}</div>
                                                </div>
                                                <div className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm">
                                                    <div className="text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Serial No.</div>
                                                    <div className="text-lg font-bold text-gray-800 font-mono">{parsedWagon.serial}</div>
                                                </div>
                                            </div>
                                        </div>
                                    ) : validDetectedNumber ? (
                                        <div className="bg-success/10 p-4 rounded-lg border border-success/30 shadow-sm">
                                            <div className="text-xs font-bold text-success uppercase tracking-wide mb-1">Confirmed Number</div>
                                            <div className="font-mono text-3xl font-bold text-gray-900 tracking-wider break-all">{validDetectedNumber}</div>
                                            <div className="flex items-center mt-2 text-success space-x-1">
                                                <CheckCircle className="w-4 h-4" />
                                                <span className="text-xs font-medium">Checksum Valid</span>
                                            </div>
                                        </div>
                                    ) : validCandidates.length > 0 ? (
                                        <div className="bg-warning/10 p-4 rounded-lg border border-warning/30 shadow-sm">
                                            <div className="text-xs font-bold text-warning uppercase tracking-wide mb-1">Potential Number</div>
                                            <div className="font-mono text-2xl font-bold text-gray-900 tracking-wider mb-2 break-all">
                                                {validCandidates[0]}
                                            </div>
                                            <div className="text-xs text-warning bg-warning/10 p-2 rounded mb-2">
                                                <span className="font-semibold">Note:</span> Number found but failed strict validation.
                                            </div>
                                            {validCandidates.length > 1 && (
                                                <div className="mt-2 text-xs text-primary">
                                                    Other candidates: {validCandidates.slice(1, 3).join(', ')}
                                                </div>
                                            )}
                                        </div>
                                    ) : validCropTexts.length > 0 ? (
                                        <div className="h-full bg-accent/10 p-4 rounded-lg border border-accent/30 shadow-sm">
                                            <div className="text-xs font-bold text-accent uppercase tracking-wide mb-1">Raw Detected Text</div>
                                            <div className="font-mono text-xl font-bold text-gray-800 break-words tracking-wide">
                                                {validCropTexts.join('  ')}
                                            </div>
                                            <div className="mt-2 flex flex-wrap gap-1">
                                                {validCropTexts.map((txt, i) => (
                                                    <span key={i} className="px-1.5 py-0.5 bg-white border border-accent/20 rounded text-xs font-mono text-accent">
                                                        {txt}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="h-full flex flex-col justify-center items-center bg-gray-50 p-6 rounded-lg border border-gray-200 text-center">
                                            <div className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2">Status</div>
                                            <div className="text-xl font-bold text-gray-500">No Number Found</div>
                                            <p className="text-sm text-gray-400 mt-2">OCR could not identify text patterns</p>
                                        </div>
                                    )}
                                </div>
                            );
                        })()}

                        {/* Damage Section - Full width for TOP, normal for SIDE */}
                        <div className={`glass-card rounded-xl p-4 h-full flex flex-col ${pipelineType === 'TOP' ? '' : ''}`}>
                            <div className="flex items-center space-x-2 mb-3">
                                <div className="bg-primary/10 p-1.5 rounded-lg border border-primary/10">
                                    <AlertCircle className="w-4 h-4 text-primary" />
                                </div>
                                <h4 className="text-sm font-bold text-slate-700 uppercase tracking-wide">Damage Detection</h4>
                                {pipelineType === 'TOP' && currentWagonStats.has_damage && (
                                    <span className="ml-auto px-2 py-1 bg-danger/10 text-danger text-xs font-medium rounded-full">
                                        Damage Found
                                    </span>
                                )}
                            </div>

                            {pipelineType === 'TOP' ? (
                                /* Enhanced TOP pipeline damage display */
                                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                    {/* Damage Count Card */}
                                    <div className={`p-4 rounded-lg ${currentWagonStats.total_detections > 0 ? 'bg-danger/10 border border-danger/30' : 'bg-success/10 border border-success/30'}`}>
                                        <div className="text-xs font-medium text-gray-500 uppercase mb-1">Damage Instances</div>
                                        <div className={`text-2xl font-bold ${currentWagonStats.total_detections > 0 ? 'text-danger' : 'text-success'}`}>
                                            {currentWagonStats.total_detections || 0}
                                        </div>
                                        {currentWagonStats.total_detections === 0 && (
                                            <div className="text-xs text-success mt-1">No damage detected</div>
                                        )}
                                    </div>

                                    {/* Avg Confidence Card */}
                                    {currentWagonStats.avg_confidence > 0 && (
                                        <div className="p-4 rounded-lg bg-warning/10 border border-warning/30">
                                            <div className="text-xs font-medium text-gray-500 uppercase mb-1">Avg Confidence</div>
                                            <div className="text-xl font-bold text-warning">
                                                {(currentWagonStats.avg_confidence * 100).toFixed(1)}%
                                            </div>
                                            <div className="text-xs text-warning mt-1">Detection confidence</div>
                                        </div>
                                    )}

                                    {/* Max Confidence Card */}
                                    {currentWagonStats.max_confidence > 0 && (
                                        <div className="p-4 rounded-lg bg-accent/10 border border-accent/30">
                                            <div className="text-xs font-medium text-gray-500 uppercase mb-1">Max Confidence</div>
                                            <div className="text-xl font-bold text-accent">
                                                {(currentWagonStats.max_confidence * 100).toFixed(1)}%
                                            </div>
                                            <div className="text-xs text-accent mt-1">Highest detection</div>
                                        </div>
                                    )}

                                    {/* Status Card */}
                                    <div className={`p-4 rounded-lg ${currentWagonStats.has_damage ? 'bg-danger/10 border border-danger/30' : 'bg-success/10 border border-success/30'}`}>
                                        <div className="text-xs font-medium text-gray-500 uppercase mb-1">Status</div>
                                        <div className={`text-xl font-bold ${currentWagonStats.has_damage ? 'text-danger' : 'text-success'}`}>
                                            {currentWagonStats.has_damage ? 'Needs Review' : 'Good'}
                                        </div>
                                        <div className={`text-xs mt-1 ${currentWagonStats.has_damage ? 'text-danger' : 'text-success'}`}>
                                            {currentWagonStats.has_damage ? 'Requires attention' : 'No issues found'}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                /* SIDE pipeline damage display */
                                <div className="flex-1">
                                    {(currentWagonStats.damage_analysis?.damage_breakdown || currentWagonStats.class_counts) &&
                                        Object.entries(currentWagonStats.damage_analysis?.damage_breakdown || currentWagonStats.class_counts || {}).filter(([_, count]) => count > 0).length > 0 ? (
                                        <div className="bg-danger/10 p-4 rounded-lg border border-danger/30 shadow-sm h-full">
                                            <div className="text-xs font-bold text-danger uppercase tracking-wide mb-3">Defects Found</div>
                                            <div className="space-y-3">
                                                {Object.entries(currentWagonStats.damage_analysis?.damage_breakdown || currentWagonStats.class_counts || {}).map(([cls, count]) => (
                                                    count > 0 && (
                                                        <div key={cls} className="flex justify-between items-center bg-white/60 p-3 rounded-lg">
                                                            <span className="text-gray-700 font-medium capitalize text-md">{cls.replace('_', ' ')}</span>
                                                            <span className="font-bold text-lg text-danger">{count}</span>
                                                        </div>
                                                    )
                                                ))}
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="bg-success/10 p-4 rounded-lg border border-success/30 shadow-sm h-full flex flex-col justify-center items-center">
                                            <div className="text-xs font-bold text-success uppercase tracking-wide mb-2">Status</div>
                                            <div className="text-xl font-bold text-success">No Damage</div>
                                            <p className="text-sm text-success mt-2">Wagon appears to be in good condition</p>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* Door Section */}
                        {pipelineType !== 'TOP' && (
                            <div className="glass-card rounded-xl p-4 h-full flex flex-col">
                                <div className="flex items-center space-x-2 mb-3">
                                    <div className="bg-primary/10 p-1.5 rounded-lg border border-primary/10">
                                        <Box className="w-4 h-4 text-primary" />
                                    </div>
                                    <h4 className="text-sm font-bold text-slate-700 uppercase tracking-wide">Doors Status</h4>
                                </div>
                                <div className="flex-1">
                                    {currentWagonDoors.door_counts && Object.entries(currentWagonDoors.door_counts).filter(([_, count]) => count > 0).length > 0 ? (
                                        <div className="bg-primary/5 p-4 rounded-lg border border-primary/20 shadow-sm h-full">
                                            <div className="text-xs font-bold text-primary uppercase tracking-wide mb-3">Door Status</div>
                                            <div className="space-y-3">
                                                {Object.entries(currentWagonDoors.door_counts).map(([status, count]) => (
                                                    count > 0 && (
                                                        <div key={status} className={`flex justify-between items-center p-3 rounded-lg ${status === 'good' ? 'bg-green-100/60' :
                                                            status === 'missing' ? 'bg-red-100/60' :
                                                                'bg-white/60'
                                                            }`}>
                                                            <span className="text-gray-700 font-medium capitalize text-md">{status}</span>
                                                            <span className={`font-bold text-lg ${status === 'good' ? 'text-green-600' :
                                                                status === 'missing' ? 'text-red-500' :
                                                                    'text-amber-600'
                                                                }`}>{count}</span>
                                                        </div>
                                                    )
                                                ))}
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 shadow-sm h-full flex flex-col justify-center items-center">
                                            <div className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2">Status</div>
                                            <div className="text-xl font-bold text-gray-500">No Door Data</div>
                                            <p className="text-sm text-gray-400 mt-2">Door detection not available</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AnalysisPage;
