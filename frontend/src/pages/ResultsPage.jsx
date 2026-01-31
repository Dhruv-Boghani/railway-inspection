import React, { useEffect, useState, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { CheckCircle, AlertTriangle, XCircle, Clock, FileText, Layers, Play, Box, Wrench, Type, Info, Download, Share2, HardDrive } from 'lucide-react';
import { useJob } from '../context/JobContext';
import NoJobFound from '../components/NoJobFound';

const ResultsPage = () => {
    const { jobId } = useParams();
    const navigate = useNavigate();
    const { setCurrentJobId } = useJob();
    const [job, setJob] = useState(null);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(true);
    const [notFound, setNotFound] = useState(false);
    const backendUrl = import.meta.env.VITE_BACKEND_URL;

    useEffect(() => {
        if (!jobId) {
            setNotFound(true);
            setLoading(false);
            return;
        }
        if (jobId) setCurrentJobId(jobId);
        const fetchData = async () => {
            try {
                const jobRes = await axios.get(`${backendUrl}/jobs/${jobId}`);
                setJob(jobRes.data);

                const resultRes = await axios.get(`${backendUrl}/jobs/${jobId}/result`);
                setResults(resultRes.data);
            } catch (err) {
                console.error("Error fetching results", err);
                if (err.response && err.response.status === 404) {
                    setNotFound(true);
                }
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [jobId]);

    if (notFound) return <NoJobFound />;
    if (loading) return <div className="p-10 text-center">Loading results...</div>;
    if (!results) return <NoJobFound />;

    // DEBUG v2 - timestamp: 6:25AM to verify hot reload is working
    console.log('[ResultsPage v2 6:25AM] FULL RESULTS:', results);
    console.log('[ResultsPage v2] results.pipeline_type:', results.pipeline_type);
    console.log('[ResultsPage v2] results.summary?.pipeline_info?.pipeline_type:', results.summary?.pipeline_info?.pipeline_type);

    // Detect TOP pipeline: check multiple possible locations for pipeline_type
    const stage4DamageCount = Object.keys(results.stage4_damage || {}).length;
    const stage5DamageCount = Object.keys(results.summary?.stage5_damage || results.stage5_damage || {}).length;
    const explicitPipelineType = results.pipeline_type || results.summary?.pipeline_info?.pipeline_type;
    const isTopPipeline = explicitPipelineType === 'TOP' || (stage4DamageCount > 0 && stage5DamageCount === 0);

    console.log('[ResultsPage v2] stage4DamageCount:', stage4DamageCount);
    console.log('[ResultsPage v2] stage5DamageCount:', stage5DamageCount);
    console.log('[ResultsPage v2] isTopPipeline:', isTopPipeline);

    const pipelineType = isTopPipeline ? 'TOP' : 'SIDE';
    const cameraAngle = isTopPipeline ? 'TOP' : (job?.camera_angle || results.summary?.pipeline_info?.camera_angle || 'LR');

    // Calculate comprehensive statistics
    // SIDE pipeline: data is at root level (pipeline_info, stage4_doors, stage5_damage)
    // TOP pipeline: data is in summary wrapper
    const totalWagons = results.pipeline_info?.total_wagons ||
        results.summary?.pipeline_info?.total_wagons ||
        (isTopPipeline ? Object.keys(results.stage4_damage || {}).length : Object.keys(results.stage5_damage || {}).length);
    const totalTime = results.pipeline_info?.total_time || results.summary?.pipeline_info?.total_time || 0;
    const stageTimes = results.pipeline_info?.stage_times || results.summary?.pipeline_info?.stage_times || {};

    // Damage data - handle both TOP and SIDE pipelines
    // TOP: results.stage4_damage
    // SIDE: results.stage5_damage (at root) or results.summary.stage5_damage
    const damageData = isTopPipeline
        ? (results.stage4_damage || {})
        : (results.stage5_damage || results.summary?.stage5_damage || {});
    // Door data - SIDE only (at root or in summary)
    const doorData = results.stage4_doors || results.summary?.stage4_doors || {};
    // OCR data - SIDE only (at root or in summary)
    const ocrData = results.stage6_ocr?.results || results.stage6_ocr || results.summary?.stage6_ocr?.results || results.summary?.stage6_ocr || [];

    // DEBUG: Log final computed values
    console.log('[ResultsPage] COMPUTED: isTopPipeline:', isTopPipeline);
    console.log('[ResultsPage] COMPUTED: pipelineType:', pipelineType);
    console.log('[ResultsPage] COMPUTED: totalWagons:', totalWagons);
    console.log('[ResultsPage] COMPUTED: damageData keys count:', Object.keys(damageData).length);

    // Door Statistics
    let totalDoorsDetected = 0;
    let goodDoors = 0;
    let damagedDoors = 0;
    let missingDoors = 0;

    Object.values(doorData).forEach(wagon => {
        if (!wagon || typeof wagon !== 'object') return;
        const counts = wagon.door_counts || {};
        const total = wagon.total_doors_detected || 0;
        totalDoorsDetected += total;

        goodDoors += counts.good || 0;
        damagedDoors += counts.damaged || 0;
        missingDoors += counts.missing || 0;
    });

    // Damage Statistics - Handle both TOP and SIDE pipeline structures
    let totalDamageWagons = 0;
    let totalDents = 0;
    let totalScratches = 0;
    let totalCracks = 0;
    let goodConditionWagons = 0;
    let totalDefectCount = 0;  // For TOP pipeline (no breakdown)

    Object.values(damageData).forEach(wagon => {
        if (!wagon || typeof wagon !== 'object') return;
        const breakdown = wagon.damage_analysis?.damage_breakdown || {};
        const totalDetections = wagon.total_detections || 0;

        if (totalDetections > 0) {
            totalDamageWagons++;
            totalDents += breakdown.dent || 0;
            totalScratches += breakdown.scratch || 0;
            totalCracks += breakdown.crack || 0;
            totalDefectCount += totalDetections;  // For TOP pipeline
        } else {
            goodConditionWagons++;
        }
    });

    // OCR Statistics - Count wagons where OCR was attempted (has crop_texts)
    // Handle both object and array structures
    const ocrDataArray = Array.isArray(ocrData) ? ocrData : Object.values(ocrData);
    const totalOCRDetected = ocrDataArray.filter(w => w && w.crop_texts && w.crop_texts.length > 0).length;
    const ocrSuccessRate = totalWagons > 0 ? ((totalOCRDetected / totalWagons) * 100).toFixed(1) : '0';

    // Chart Data - For TOP pipeline, show total defects if no breakdown
    const damageChartData = pipelineType === 'TOP' && totalDefectCount > 0 && (totalDents + totalScratches + totalCracks === 0)
        ? [{ name: 'Defects', value: totalDefectCount, color: '#8b5cf6' }]
        : [
            { name: 'Dents', value: totalDents, color: '#4f46e5' },
            { name: 'Scratches', value: totalScratches, color: '#8b5cf6' },
            { name: 'Cracks', value: totalCracks, color: '#a5b4fc' },
        ].filter(d => d.value > 0);

    const doorChartData = [
        { name: 'Good', value: goodDoors, color: '#a5b4fc', fill: '#a5b4fc' },
        { name: 'Damaged', value: damagedDoors, color: '#8b5cf6', fill: '#8b5cf6' },
        { name: 'Missing', value: missingDoors, color: '#4f46e5', fill: '#4f46e5' },
    ].filter(d => d.value > 0);

    const healthDistribution = [
        { name: 'Good', value: goodConditionWagons, color: '#a5b4fc' },
        { name: 'With Damage', value: totalDamageWagons, color: '#8b5cf6' },
    ];

    // Stage Processing Time Chart Data
    const stageTimeChartData = Object.entries(stageTimes).map(([stage, time]) => {
        // Format stage names nicely
        const stageNames = {
            'stage1': 'Wagon Detection',
            'stage2': 'Quality Assessment',
            'stage3': 'Image Enhancement',
            'stage4': pipelineType === 'TOP' ? 'Damage Detection' : 'Door Detection',
            'stage5': 'Damage Detection',
            'stage6': 'OCR Extraction',
        };
        return {
            name: stageNames[stage] || stage,
            time: parseFloat(time.toFixed(2)),
            color: '#8b5cf6',
        };
    }).filter(d => d.time > 0);

    // Get video URL
    const jobDirName = job?.output_dir?.split('\\').pop().split('/').pop();
    const videoUrl = jobDirName ? `${backendUrl}/outputs/${jobDirName}/stage1_annotated.mp4` : null;

    // Placeholder for wagonData, assuming it would be derived from results if the table was fully implemented
    const wagonData = {}; // This would need to be populated if the table is fully functional

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-3xl font-bold text-slate-800 tracking-tight">Inspection Results</h1>
                    <p className="text-slate-500 mt-1 font-medium">Detailed breakdown of current session analysis</p>
                </div>
                <div className="flex gap-3">
                    <button className="flex items-center gap-2 px-4 py-2 bg-white text-slate-600 rounded-xl shadow-sm border border-slate-200 hover:bg-slate-50 transition-colors font-medium">
                        <Download className="w-4 h-4" />
                        <span>Export PDF</span>
                    </button>
                    <button className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-xl shadow-lg shadow-indigo-500/30 hover:bg-primary-dark transition-all transform hover:-translate-y-0.5 font-medium">
                        <Share2 className="w-4 h-4" />
                        <span>Share Report</span>
                    </button>
                </div>
            </div>

            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="glass-card hover:translate-y-[-2px] hover:shadow-xl transition-all p-5 rounded-2xl">
                    <div className="flex items-center justify-between mb-2">
                        <div className="bg-indigo-500/10 p-2 rounded-xl backdrop-blur-sm">
                            <Layers className="w-6 h-6 text-primary" />
                        </div>
                        <span className="text-3xl font-bold text-slate-800">{totalWagons}</span>
                    </div>
                    <p className="text-indigo-900/40 text-sm font-bold uppercase tracking-wider">Total Wagons</p>
                </div>

                <div className="glass-card hover:translate-y-[-2px] hover:shadow-xl transition-all p-5 rounded-2xl">
                    <div className="flex items-center justify-between mb-2">
                        <div className="bg-emerald-500/10 p-2 rounded-xl backdrop-blur-sm">
                            <CheckCircle className="w-6 h-6 text-emerald-500" />
                        </div>
                        <span className="text-3xl font-bold text-slate-800">{goodConditionWagons}</span>
                    </div>
                    <p className="text-emerald-900/40 text-sm font-bold uppercase tracking-wider">Good Condition</p>
                </div>

                <div className="glass-card hover:translate-y-[-2px] hover:shadow-xl transition-all p-5 rounded-2xl">
                    <div className="flex items-center justify-between mb-2">
                        <div className="bg-red-500/10 p-2 rounded-xl backdrop-blur-sm">
                            <AlertTriangle className="w-6 h-6 text-red-500" />
                        </div>
                        <span className="text-3xl font-bold text-slate-800">{totalDamageWagons}</span>
                    </div>
                    <p className="text-red-900/40 text-sm font-bold uppercase tracking-wider">With Damage</p>
                </div>

                <div className="glass-card hover:translate-y-[-2px] hover:shadow-xl transition-all p-5 rounded-2xl">
                    <div className="flex items-center justify-between mb-2">
                        <div className="bg-purple-500/10 p-2 rounded-xl backdrop-blur-sm">
                            <Clock className="w-6 h-6 text-accent" />
                        </div>
                        <span className="text-3xl font-bold text-slate-800">{totalTime.toFixed(1)}s</span>
                    </div>
                    <p className="text-purple-900/40 text-sm font-bold uppercase tracking-wider">Processing Time</p>
                </div>
            </div>

            {/* Video and Quick Stats */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Annotated Video */}
                <div className="lg:col-span-2 glass-panel rounded-2xl overflow-hidden">
                    <div className="p-4 border-b border-white/40 bg-white/40 backdrop-blur-sm">
                        <div className="flex items-center space-x-2">
                            <Play className="w-5 h-5 text-primary" />
                            <h3 className="font-bold text-slate-800">Annotated Inspection Video</h3>
                        </div>
                    </div>
                    <div className="p-4 bg-slate-900 flex flex-col items-center justify-center" style={{ minHeight: '400px' }}>
                        {videoUrl ? (
                            <>
                                <video
                                    key={videoUrl}
                                    controls
                                    preload="metadata"
                                    className="w-full h-full max-h-[500px] rounded mb-3"
                                    playsInline
                                >
                                    <source src={videoUrl} type="video/mp4" />
                                    Your browser does not support the video tag.
                                </video>
                            </>
                        ) : (
                            <div className="text-gray-400 text-center">
                                <Play className="w-16 h-16 mx-auto mb-2 opacity-50" />
                                <p>No video available</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Quick Stats Panel */}
                <div className="glass-panel rounded-xl p-6 flex flex-col">
                    <h3 className="font-bold text-gray-800 mb-4 flex items-center space-x-2">
                        <Info className="w-5 h-5 text-primary" />
                        <span>Quick Stats</span>
                    </h3>
                    <div className="flex-1 flex flex-col justify-between space-y-4">
                        {/* OCR Success - Only for SIDE view */}
                        {cameraAngle !== 'TOP' && (
                            <div className="glass-card p-5 rounded-lg flex flex-col justify-center">
                                <div className="flex items-center justify-between mb-3">
                                    <span className="text-base font-medium text-gray-700 flex items-center space-x-2">
                                        <Type className="w-5 h-5 text-primary" />
                                        <span>OCR Detected</span>
                                    </span>
                                    <span className="text-2xl font-bold text-primary">{totalOCRDetected}/{totalWagons}</span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-3">
                                    <div className="bg-accent h-3 rounded-full" style={{ width: `${ocrSuccessRate}%` }}></div>
                                </div>
                                <span className="text-sm text-gray-500 font-medium mt-2 block text-center">{ocrSuccessRate}% Success</span>
                            </div>
                        )}

                        {/* Total Doors - Only for SIDE view */}
                        {cameraAngle !== 'TOP' && (
                            <div className="glass-card p-5 rounded-lg flex flex-col justify-center">
                                <div className="flex items-center justify-between mb-3">
                                    <span className="text-base font-medium text-gray-700 flex items-center space-x-2">
                                        <Box className="w-5 h-5 text-primary" />
                                        <span>Total Doors</span>
                                    </span>
                                    <span className="text-2xl font-bold text-primary">{totalDoorsDetected}</span>
                                </div>
                                <div className="grid grid-cols-3 gap-3">
                                    <div className="text-center p-3 bg-white/50 rounded-lg border border-gray-200">
                                        <div className="text-xl font-bold text-success">{goodDoors}</div>
                                        <div className="text-sm text-gray-500 mt-1">Good</div>
                                    </div>
                                    <div className="text-center p-3 bg-white/50 rounded-lg border border-gray-200">
                                        <div className="text-xl font-bold text-warning">{damagedDoors}</div>
                                        <div className="text-sm text-gray-500 mt-1">Damaged</div>
                                    </div>
                                    <div className="text-center p-3 bg-white/50 rounded-lg border border-gray-200">
                                        <div className="text-xl font-bold text-danger">{missingDoors}</div>
                                        <div className="text-sm text-gray-500 mt-1">Missing</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Floor Damage (TOP) / Total Defects (SIDE) */}
                        <div className="glass-card p-5 rounded-lg flex flex-col justify-center">
                            <div className="flex items-center justify-between mb-4">
                                <span className="text-base font-medium text-gray-700 flex items-center space-x-2">
                                    <Wrench className="w-5 h-5 text-primary" />
                                    <span>{pipelineType === 'TOP' ? 'Floor Damage' : 'Total Defects'}</span>
                                </span>
                                <span className="text-2xl font-bold text-primary">
                                    {pipelineType === 'TOP' ? totalDefectCount : (totalDents + totalScratches + totalCracks)}
                                </span>
                            </div>
                            {pipelineType === 'TOP' ? (
                                <div className="space-y-4">
                                    <div className="text-center p-6 bg-white/50 rounded-xl border border-gray-200">
                                        <div className="text-4xl font-bold text-accent mb-2">{totalDefectCount}</div>
                                        <div className="text-sm text-gray-500 font-medium">Floor Damage Detected</div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="text-center p-3 bg-white/50 rounded-lg border border-gray-200">
                                            <div className="text-xl font-bold text-danger">{totalDamageWagons}</div>
                                            <div className="text-xs text-gray-500">Wagons Affected</div>
                                        </div>
                                        <div className="text-center p-3 bg-white/50 rounded-lg border border-gray-200">
                                            <div className="text-xl font-bold text-success">{goodConditionWagons}</div>
                                            <div className="text-xs text-gray-500">Good Condition</div>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="text-center p-3 bg-white/50 rounded-lg border border-gray-200">
                                        <div className="text-xl font-bold text-danger">{totalDents}</div>
                                        <div className="text-sm text-gray-500 mt-1">Dents</div>
                                    </div>
                                    <div className="text-center p-3 bg-white/50 rounded-lg border border-gray-200">
                                        <div className="text-xl font-bold text-warning">{totalScratches}</div>
                                        <div className="text-sm text-gray-500 mt-1">Scratches</div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Charts Section - Only for SIDE view */}
            {pipelineType !== 'TOP' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Health Distribution */}
                    <div className="glass-panel p-6 rounded-xl">
                        <h3 className="text-lg font-bold text-gray-800 mb-4">Wagon Health</h3>
                        <div className="h-[250px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={healthDistribution}
                                        innerRadius={50}
                                        outerRadius={80}
                                        paddingAngle={3}
                                        dataKey="value"
                                        label={({ name, value, percent }) => `${name}: ${value}`}
                                        labelLine={false}
                                    >
                                        {healthDistribution.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip formatter={(value, name) => [`${value} wagons`, name]} />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                        {/* Legend with values */}
                        <div className="flex justify-center gap-6 mt-2">
                            {healthDistribution.map((entry, index) => (
                                <div key={index} className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }}></div>
                                    <span className="text-sm text-gray-600">{entry.name}: <strong>{entry.value}</strong></span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Damage Breakdown */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                        <h3 className="text-lg font-bold text-gray-800 mb-4">Damage Types</h3>
                        <div className="h-[250px]">
                            {damageChartData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={damageChartData}>
                                        <XAxis dataKey="name" />
                                        <YAxis />
                                        <Tooltip formatter={(value) => [`${value} occurrences`]} />
                                        <Bar dataKey="value" radius={[8, 8, 0, 0]} label={{ position: 'top', fill: '#374151', fontWeight: 'bold' }}>
                                            {damageChartData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="h-full flex items-center justify-center text-gray-400">
                                    <div className="text-center">
                                        <CheckCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                        <p className="text-sm">No damage detected</p>
                                    </div>
                                </div>
                            )}
                        </div>
                        {/* Legend with values */}
                        {damageChartData.length > 0 && (
                            <div className="flex justify-center gap-4 mt-2">
                                {damageChartData.map((entry, index) => (
                                    <div key={index} className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded" style={{ backgroundColor: entry.color }}></div>
                                        <span className="text-sm text-gray-600">{entry.name}: <strong>{entry.value}</strong></span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Door Status */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                        <h3 className="text-lg font-bold text-gray-800 mb-4">Door Status</h3>
                        <div className="h-[250px]">
                            {doorChartData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={doorChartData}
                                            cx="50%"
                                            cy="50%"
                                            outerRadius={70}
                                            dataKey="value"
                                            label={({ name, value }) => `${name}: ${value}`}
                                            labelLine={false}
                                        >
                                            {doorChartData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Pie>
                                        <Tooltip formatter={(value, name) => [`${value} doors`, name]} />
                                    </PieChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="h-full flex items-center justify-center text-gray-400">
                                    <div className="text-center">
                                        <Box className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                        <p className="text-sm">No door data</p>
                                    </div>
                                </div>
                            )}
                        </div>
                        {/* Legend with values */}
                        {doorChartData.length > 0 && (
                            <div className="flex justify-center gap-4 mt-2">
                                {doorChartData.map((entry, index) => (
                                    <div key={index} className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }}></div>
                                        <span className="text-sm text-gray-600">{entry.name}: <strong>{entry.value}</strong></span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Stage Processing Time - Full Width (SIDE pipeline only) */}
            {!isTopPipeline && (
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h3 className="font-bold text-lg text-gray-800 flex items-center gap-2">
                                <Clock className="w-6 h-6 text-primary" />
                                Stage Processing Time
                            </h3>
                            <p className="text-sm text-gray-500 mt-1">Time taken by each pipeline stage</p>
                        </div>
                        <div className="text-right">
                            <div className="text-2xl font-bold text-primary">{totalTime.toFixed(2)}s</div>
                            <div className="text-xs text-gray-400">Total Processing Time</div>
                        </div>
                    </div>
                    <div className="h-[280px]">
                        {stageTimeChartData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={stageTimeChartData} layout="vertical" margin={{ top: 5, right: 60, left: 10, bottom: 5 }}>
                                    <XAxis type="number" unit="s" tick={{ fontSize: 12 }} axisLine={{ stroke: '#E5E7EB' }} tickLine={{ stroke: '#E5E7EB' }} />
                                    <YAxis
                                        dataKey="name"
                                        type="category"
                                        width={150}
                                        tick={{ fontSize: 13, fill: '#374151', fontWeight: 500 }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip
                                        formatter={(value) => [`${value} seconds`, 'Processing Time']}
                                        contentStyle={{ borderRadius: '8px', border: '1px solid #E5E7EB' }}
                                    />
                                    <Bar
                                        dataKey="time"
                                        radius={[0, 8, 8, 0]}
                                        barSize={35}
                                        label={{
                                            position: 'right',
                                            fill: '#374151',
                                            fontSize: 13,
                                            fontWeight: 600,
                                            formatter: (v) => `${v}s`
                                        }}
                                    >
                                        {stageTimeChartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#7C6CF2' : '#4C4FB3'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex items-center justify-center text-gray-400">
                                <div className="text-center">
                                    <Clock className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                    <p className="text-sm">No timing data available</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ResultsPage;
