import React, { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FileText, Download, Printer, ArrowLeft, CheckCircle, AlertTriangle, XCircle, Loader2 } from 'lucide-react';
import axios from 'axios';
import html2pdf from 'html2pdf.js';
import NoJobFound from '../components/NoJobFound';

const ReportDetailPage = () => {
    const { jobId } = useParams();
    const navigate = useNavigate();
    const [results, setResults] = useState(null);
    const [job, setJob] = useState(null);
    const [loading, setLoading] = useState(true);
    const [notFound, setNotFound] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);
    const reportRef = useRef(null);

    useEffect(() => {
        if (!jobId) {
            setNotFound(true);
            setLoading(false);
            return;
        }
        const fetchData = async () => {
            try {
                const backendUrl = import.meta.env.VITE_BACKEND_URL;
                const [jobRes, resultRes] = await Promise.all([
                    axios.get(`${backendUrl}/jobs/${jobId}`),
                    axios.get(`${backendUrl}/jobs/${jobId}/result`)
                ]);
                setJob(jobRes.data);
                setResults(resultRes.data);
            } catch (error) {
                console.error('Error fetching data:', error);
                if (error.response && error.response.status === 404) {
                    setNotFound(true);
                }
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [jobId]);

    const handlePrint = () => {
        window.print();
    };

    const handleDownload = async () => {
        if (!reportRef.current) return;

        setIsDownloading(true);

        const opt = {
            margin: [10, 10, 10, 10],
            filename: `Railway_Inspection_Report_${jobId}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: {
                scale: 2,
                useCORS: true,
                logging: false
            },
            jsPDF: {
                unit: 'mm',
                format: 'a4',
                orientation: 'portrait'
            },
            pagebreak: { mode: 'avoid-all' }
        };

        try {
            await html2pdf().set(opt).from(reportRef.current).save();
        } catch (error) {
            console.error('Error generating PDF:', error);
            alert('Failed to generate PDF. Please try again.');
        } finally {
            setIsDownloading(false);
        }
    };

    if (notFound) {
        return <NoJobFound />;
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
                    <p className="mt-4 text-gray-600">Loading report...</p>
                </div>
            </div>
        );
    }

    if (!results || !job) {
        return <NoJobFound />;
    }

    // SIDE pipeline: data at root level, TOP: stage4_damage at root with summary wrapper
    const totalWagons = results.pipeline_info?.total_wagons || results.summary?.pipeline_info?.total_wagons || 0;
    const totalTime = results.pipeline_info?.total_time || results.summary?.pipeline_info?.total_time || 0;

    // Detect pipeline type
    const pipelineType = results.pipeline_type || 'SIDE';

    // Get data based on pipeline type - check root first for SIDE
    const doorData = results.stage4_doors || results.summary?.stage4_doors || {};
    const damageData = pipelineType === 'TOP'
        ? (results.stage4_damage || {})
        : (results.stage5_damage || results.summary?.stage5_damage || {});
    const ocrData = results.stage6_ocr?.results || results.stage6_ocr || results.summary?.stage6_ocr?.results || results.summary?.stage6_ocr || [];

    let goodDoors = 0, damagedDoors = 0, missingDoors = 0, totalDoorsDetected = 0;
    Object.values(doorData).forEach(wagon => {
        if (!wagon || typeof wagon !== 'object') return;
        const counts = wagon.door_counts || {};
        totalDoorsDetected += wagon.total_doors_detected || 0;
        goodDoors += counts.good || 0;
        damagedDoors += counts.damaged || 0;
        missingDoors += counts.missing || 0;
    });

    // Calculate damage stats - handle both TOP and SIDE data structures
    let totalDents = 0, totalScratches = 0, totalCracks = 0, totalFloorDamage = 0;
    let goodConditionWagons = 0, totalDamageWagons = 0;
    Object.values(damageData).forEach(wagon => {
        if (!wagon || typeof wagon !== 'object') return;

        if (pipelineType === 'TOP') {
            // TOP pipeline: simple total_detections count
            const totalDetections = wagon.total_detections || 0;
            if (totalDetections > 0) {
                totalDamageWagons++;
                totalFloorDamage += totalDetections;
            } else {
                goodConditionWagons++;
            }
        } else {
            // SIDE pipeline: damage_analysis.damage_breakdown
            const breakdown = wagon.damage_analysis?.damage_breakdown || {};
            const totalDetections = wagon.total_detections || 0;

            if (totalDetections > 0) {
                totalDamageWagons++;
                totalDents += breakdown.dent || 0;
                totalScratches += breakdown.scratch || 0;
                totalCracks += breakdown.crack || 0;
            } else {
                goodConditionWagons++;
            }
        }
    });

    const ocrDataArray = Array.isArray(ocrData) ? ocrData : [];
    // Fix: Check crop_texts array for actual OCR detections (same as ResultsPage)
    const totalOCRDetected = ocrDataArray.filter(w => w && w.crop_texts && w.crop_texts.length > 0).length;
    const ocrSuccessRate = totalWagons > 0 ? ((totalOCRDetected / totalWagons) * 100).toFixed(1) : '0';

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Print/Download Header - Hidden when printing */}
            <div className="print:hidden bg-white border-b border-gray-200 px-8 py-4 sticky top-0 z-10">
                <div className="flex justify-between items-center">
                    <button
                        onClick={() => navigate('/reports')}
                        className="flex items-center space-x-2 text-gray-600 hover:text-gray-900"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back to Reports</span>
                    </button>
                    <div className="flex space-x-3">
                        <button
                            onClick={handlePrint}
                            className="bg-white border border-gray-200 text-gray-700 px-5 py-2.5 rounded-lg font-medium hover:bg-gray-50 flex items-center space-x-2"
                        >
                            <Printer className="w-4 h-4" />
                            <span>Print Report</span>
                        </button>
                        <button
                            onClick={handleDownload}
                            disabled={isDownloading}
                            className="bg-primary text-white px-5 py-2.5 rounded-lg font-medium hover:bg-primary-light flex items-center space-x-2 disabled:opacity-70 disabled:cursor-not-allowed"
                        >
                            {isDownloading ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span>Generating...</span>
                                </>
                            ) : (
                                <>
                                    <Download className="w-4 h-4" />
                                    <span>Download PDF</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Report Content */}
            <div ref={reportRef} className="max-w-5xl mx-auto px-8 py-8 print:px-0 bg-gray-50">
                {/* Report Header */}
                <div className="bg-white rounded-lg shadow-sm p-8 mb-6 print:shadow-none">
                    <div className="flex justify-between items-start mb-6">
                        <div>
                            <h1 className="text-3xl font-bold text-primary mb-2">Railway Wagon Inspection Report</h1>
                            <p className="text-gray-600">Comprehensive Analysis Report</p>
                        </div>
                        <div className="text-right">
                            <p className="text-sm text-gray-500">Report ID</p>
                            <p className="font-mono font-semibold">#{jobId}</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-6 pt-6 border-t border-gray-200">
                        <div>
                            <p className="text-sm text-gray-500">Inspection Date</p>
                            <p className="font-semibold">{new Date().toLocaleDateString()}</p>
                        </div>
                        <div>
                            <p className="text-sm text-gray-500">Total Wagons</p>
                            <p className="font-semibold">{totalWagons}</p>
                        </div>
                        <div>
                            <p className="text-sm text-gray-500">Processing Time</p>
                            <p className="font-semibold">{totalTime.toFixed(1)}s</p>
                        </div>
                    </div>
                </div>

                {/* Executive Summary */}
                <div className="bg-white rounded-lg shadow-sm p-8 mb-6 print:shadow-none print:break-inside-avoid">
                    <h2 className="text-2xl font-bold text-gray-800 mb-4">Executive Summary</h2>
                    <div className="grid grid-cols-2 gap-6">
                        <div className="p-4 bg-green-50 rounded-lg border border-green-100">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-gray-600">Good Condition</p>
                                    <p className="text-3xl font-bold text-green-600">{goodConditionWagons}</p>
                                </div>
                                <CheckCircle className="w-12 h-12 text-green-600 opacity-50" />
                            </div>
                        </div>
                        <div className="p-4 bg-red-50 rounded-lg border border-red-100">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-gray-600">With Damage</p>
                                    <p className="text-3xl font-bold text-red-600">{totalDamageWagons}</p>
                                </div>
                                <AlertTriangle className="w-12 h-12 text-red-600 opacity-50" />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Detailed Statistics */}
                <div className="bg-white rounded-lg shadow-sm p-8 mb-6 print:shadow-none print:break-inside-avoid">
                    <h2 className="text-2xl font-bold text-gray-800 mb-6">Detailed Statistics</h2>

                    {/* Door Analysis - Only for SIDE pipeline */}
                    {pipelineType !== 'TOP' && (
                        <div className="mb-6">
                            <h3 className="text-lg font-semibold text-gray-700 mb-3">Door Detection Analysis</h3>
                            <div className="grid grid-cols-4 gap-4">
                                <div className="p-4 bg-amber-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-amber-600">{totalDoorsDetected}</p>
                                    <p className="text-sm text-gray-600 mt-1">Total Doors</p>
                                </div>
                                <div className="p-4 bg-green-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-green-600">{goodDoors}</p>
                                    <p className="text-sm text-gray-600 mt-1">Good Doors</p>
                                </div>
                                <div className="p-4 bg-orange-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-orange-600">{damagedDoors}</p>
                                    <p className="text-sm text-gray-600 mt-1">Damaged</p>
                                </div>
                                <div className="p-4 bg-red-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-red-600">{missingDoors}</p>
                                    <p className="text-sm text-gray-600 mt-1">Missing</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Floor Damage Analysis - Only for TOP pipeline */}
                    {pipelineType === 'TOP' && (
                        <div className="mb-6">
                            <h3 className="text-lg font-semibold text-gray-700 mb-3">Floor Damage Analysis</h3>
                            <div className="grid grid-cols-3 gap-4">
                                <div className="p-4 bg-red-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-red-600">{totalFloorDamage}</p>
                                    <p className="text-sm text-gray-600 mt-1">Total Floor Damage</p>
                                </div>
                                <div className="p-4 bg-orange-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-orange-600">{totalDamageWagons}</p>
                                    <p className="text-sm text-gray-600 mt-1">Wagons with Damage</p>
                                </div>
                                <div className="p-4 bg-green-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-green-600">{goodConditionWagons}</p>
                                    <p className="text-sm text-gray-600 mt-1">Good Condition</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Damage Analysis - Only for SIDE pipeline */}
                    {pipelineType !== 'TOP' && (
                        <div className="mb-6">
                            <h3 className="text-lg font-semibold text-gray-700 mb-3">Damage Detection Analysis</h3>
                            <div className="grid grid-cols-4 gap-4">
                                <div className="p-4 bg-red-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-red-600">{totalDents + totalScratches + totalCracks}</p>
                                    <p className="text-sm text-gray-600 mt-1">Total Defects</p>
                                </div>
                                <div className="p-4 bg-rose-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-rose-600">{totalDents}</p>
                                    <p className="text-sm text-gray-600 mt-1">Dents</p>
                                </div>
                                <div className="p-4 bg-orange-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-orange-600">{totalScratches}</p>
                                    <p className="text-sm text-gray-600 mt-1">Scratches</p>
                                </div>
                                <div className="p-4 bg-amber-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-amber-600">{totalCracks}</p>
                                    <p className="text-sm text-gray-600 mt-1">Cracks</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* OCR Analysis - Only for SIDE pipeline */}
                    {pipelineType !== 'TOP' && (
                        <div>
                            <h3 className="text-lg font-semibold text-gray-700 mb-3">OCR Detection Analysis</h3>
                            <div className="grid grid-cols-3 gap-4">
                                <div className="p-4 bg-indigo-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-indigo-600">{totalOCRDetected}</p>
                                    <p className="text-sm text-gray-600 mt-1">Detected</p>
                                </div>
                                <div className="p-4 bg-purple-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-purple-600">{totalWagons - totalOCRDetected}</p>
                                    <p className="text-sm text-gray-600 mt-1">Not Detected</p>
                                </div>
                                <div className="p-4 bg-blue-50 rounded-lg text-center">
                                    <p className="text-2xl font-bold text-blue-600">{ocrSuccessRate}%</p>
                                    <p className="text-sm text-gray-600 mt-1">Success Rate</p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="text-center text-gray-500 text-sm mt-8 print:mt-12">
                    <p>© 2026 Railway Inspector - Automated Wagon Inspection System</p>
                    <p className="mt-1">Generated on {new Date().toLocaleString()}</p>
                </div>
            </div>
        </div>
    );
};

export default ReportDetailPage;
