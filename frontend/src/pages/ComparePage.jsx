import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, ArrowRight, Split, Eye, AlertTriangle, CheckCircle, XCircle, Loader2, DoorOpen, Wrench, Type, Train } from 'lucide-react';

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
    return 'Unknown';
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

const ComparePage = () => {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [loadingStatus, setLoadingStatus] = useState('');
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const f = e.target.files[0];
        if (f) {
            setFile(f);
            setPreviewUrl(URL.createObjectURL(f));
            setResult(null);
        }
    };

    const handleCompare = async () => {
        if (!file) return;
        setLoading(true);
        setLoadingStatus('Uploading frame...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            setLoadingStatus('Running pipeline on blur frame...');
            const res = await axios.post('http://localhost:8000/tools/compare-pipeline', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 300000 // 5 minutes timeout for long-running pipeline
            });
            setResult(res.data);
        } catch (err) {
            console.error(err);
            alert("Comparison failed: " + (err.response?.data?.detail || err.message));
        } finally {
            setLoading(false);
            setLoadingStatus('');
        }
    };

    const handleClear = () => {
        setFile(null);
        setPreviewUrl(null);
        setResult(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    // Helper to render door counts with colors
    const DoorCountBadge = ({ counts }) => (
        <div className="flex gap-3 flex-wrap">
            <span className="px-3 py-1.5 bg-success/10 text-success border border-success/30 rounded-lg text-sm font-semibold">
                Good: {counts?.good || 0}
            </span>
            <span className="px-3 py-1.5 bg-warning/10 text-warning border border-warning/30 rounded-lg text-sm font-semibold">
                Damaged: {counts?.damaged || 0}
            </span>
            <span className="px-3 py-1.5 bg-danger/10 text-danger border border-danger/30 rounded-lg text-sm font-semibold">
                Missing: {counts?.missing || 0}
            </span>
        </div>
    );

    // Helper to render severity badge  
    const SeverityBadge = ({ severity }) => {
        const colors = {
            none: 'bg-green-100 text-green-700',
            minor: 'bg-yellow-100 text-yellow-700',
            moderate: 'bg-amber-100 text-amber-700',
            severe: 'bg-red-100 text-red-700',
            unknown: 'bg-gray-100 text-gray-600'
        };
        return (
            <span className={`px-3 py-1.5 rounded-lg text-sm font-semibold ${colors[severity] || colors.unknown}`}>
                {severity?.toUpperCase() || 'UNKNOWN'}
            </span>
        );
    };

    // Render results card for one side (blur or deblur)
    const ResultCard = ({ title, data, isDeblur = false }) => {
        if (!data) return null;

        const doorResults = data.door_results || {};
        const damageResults = data.damage_results || {};
        const ocrResults = data.ocr_results || {};

        return (
            <div className={`bg-white rounded-xl shadow-sm border ${isDeblur ? 'border-accent/30' : 'border-gray-200'} overflow-hidden`}>
                {/* Header */}
                <div className={`px-4 py-3 ${isDeblur ? 'bg-accent/10' : 'bg-gray-50'} border-b`}>
                    <h3 className={`font-bold ${isDeblur ? 'text-accent' : 'text-gray-700'}`}>
                        {title}
                    </h3>
                </div>

                {/* Main Image */}
                <div className="p-4">
                    {data.image_url && (
                        <img
                            src={`http://localhost:8000${data.image_url}`}
                            alt={title}
                            className="w-full h-48 object-contain bg-gray-100 rounded-lg mb-4"
                        />
                    )}

                    {/* Door Detection */}
                    <div className="mb-5 p-4 bg-gray-50 rounded-xl border border-gray-200">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="bg-primary/10 p-2 rounded-lg">
                                <DoorOpen className="w-5 h-5 text-primary" />
                            </div>
                            <span className="font-semibold text-base text-gray-800">Door Detection</span>
                            <span className="ml-auto text-sm font-medium text-gray-500 bg-white px-2 py-1 rounded-lg border border-gray-200">
                                {doorResults.total_doors || 0} doors
                            </span>
                        </div>
                        <DoorCountBadge counts={doorResults.door_counts} />
                        {doorResults.annotated_url && (
                            <img
                                src={`http://localhost:8000${doorResults.annotated_url}`}
                                alt="Door annotations"
                                className="w-full h-48 object-contain bg-white rounded-lg mt-3 border border-gray-200"
                            />
                        )}
                        {doorResults.error && (
                            <p className="text-sm text-red-500 mt-2">{doorResults.error}</p>
                        )}
                    </div>

                    {/* Damage Detection */}
                    <div className="mb-5 p-4 bg-gray-50 rounded-xl border border-gray-200">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="bg-primary/10 p-2 rounded-lg">
                                <Wrench className="w-5 h-5 text-primary" />
                            </div>
                            <span className="font-semibold text-base text-gray-800">Damage Detection</span>
                            <span className="ml-auto text-sm font-medium text-gray-500 bg-white px-2 py-1 rounded-lg border border-gray-200">
                                {damageResults.total_detections || 0} found
                            </span>
                        </div>
                        <div className="flex items-center gap-3 mb-3">
                            <span className="text-sm text-gray-600 font-medium">Severity:</span>
                            <SeverityBadge severity={damageResults.severity} />
                            {damageResults.total_damage_percent > 0 && (
                                <span className="text-sm text-gray-500">
                                    ({damageResults.total_damage_percent}% area)
                                </span>
                            )}
                        </div>
                        {damageResults.annotated_url && (
                            <img
                                src={`http://localhost:8000${damageResults.annotated_url}`}
                                alt="Damage annotations"
                                className="w-full h-48 object-contain bg-white rounded-lg mt-3 border border-gray-200"
                            />
                        )}
                        {damageResults.error && (
                            <p className="text-sm text-red-500 mt-2">{damageResults.error}</p>
                        )}
                    </div>

                    {/* OCR */}
                    <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="bg-primary/10 p-2 rounded-lg">
                                <Type className="w-5 h-5 text-primary" />
                            </div>
                            <span className="font-semibold text-base text-gray-800">OCR (Text Detection)</span>
                            <span className="ml-auto text-sm font-medium text-gray-500 bg-white px-2 py-1 rounded-lg border border-gray-200">
                                {ocrResults.count || 0} found
                            </span>
                        </div>
                        {ocrResults.parsed_wagons?.length > 0 ? (
                            <div className="space-y-4">
                                {ocrResults.parsed_wagons.map((wagon, i) => (
                                    <div key={i} className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                                        {/* Wagon Number Header */}
                                        <div className="flex items-center gap-3 mb-4 pb-3 border-b border-gray-100">
                                            <div className="bg-primary/10 p-2 rounded-lg">
                                                <Train className="w-5 h-5 text-primary" />
                                            </div>
                                            <span className="font-mono text-xl font-bold text-primary tracking-wider">
                                                {wagon.number}
                                            </span>
                                            {wagon.check_digit_valid ? (
                                                <span className="px-2 py-1 bg-success/10 text-success border border-success/30 rounded-full text-xs font-semibold flex items-center gap-1">
                                                    <CheckCircle className="w-3 h-3" />
                                                    Valid
                                                </span>
                                            ) : (
                                                <span className="px-2 py-1 bg-danger/10 text-danger border border-danger/30 rounded-full text-xs font-semibold">
                                                    ✗ Invalid (exp: {wagon.expected_check_digit})
                                                </span>
                                            )}
                                        </div>
                                        {/* Details Grid */}
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="bg-gray-50 p-3 rounded-lg">
                                                <div className="text-xs text-gray-500 uppercase font-medium mb-1">Wagon Type</div>
                                                <div className="text-base font-semibold text-gray-800">{getWagonType(wagon.wagon_type_code)}</div>
                                                <div className="text-xs text-gray-400 mt-0.5">Code: {wagon.wagon_type_code}</div>
                                            </div>
                                            <div className="bg-gray-50 p-3 rounded-lg">
                                                <div className="text-xs text-gray-500 uppercase font-medium mb-1">Owning Railway</div>
                                                <div className="text-base font-semibold text-gray-800">{getRailwayName(wagon.owning_railway_code)}</div>
                                                <div className="text-xs text-gray-400 mt-0.5">Code: {wagon.owning_railway_code}</div>
                                            </div>
                                            <div className="bg-gray-50 p-3 rounded-lg">
                                                <div className="text-xs text-gray-500 uppercase font-medium mb-1">Year of Manufacture</div>
                                                <div className="text-base font-semibold text-gray-800">{wagon.year_of_manufacture}</div>
                                            </div>
                                            <div className="bg-gray-50 p-3 rounded-lg">
                                                <div className="text-xs text-gray-500 uppercase font-medium mb-1">Serial Number</div>
                                                <div className="text-base font-semibold text-gray-800">{wagon.individual_serial}</div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : ocrResults.detected_numbers?.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                                {ocrResults.detected_numbers.map((num, i) => (
                                    <span key={i} className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-mono">
                                        {num}
                                    </span>
                                ))}
                            </div>
                        ) : (
                            <span className="text-xs text-gray-400">No text detected</span>
                        )}
                        {ocrResults.error && (
                            <p className="text-xs text-red-500 mt-1">{ocrResults.error}</p>
                        )}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="min-h-[calc(100vh-6rem)] flex flex-col gap-6">
            {/* Header */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-start space-x-4">
                    <div className="bg-primary text-white p-3 rounded-lg">
                        <Split className="w-8 h-8" />
                    </div>
                    <div className="flex-1">
                        <h2 className="text-2xl font-bold text-primary">Blur vs Deblur Pipeline Comparison</h2>
                        <p className="text-gray-500 mt-1">
                            Upload a blur wagon frame to compare detection results before and after AI enhancement.
                        </p>
                    </div>
                </div>
            </div>






            {/* Upload Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div
                    className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                        ${file ? 'border-accent/50 bg-accent/5' : 'border-gray-300 hover:border-primary/30 hover:bg-gray-50'}`}
                    onClick={() => !file && fileInputRef.current?.click()}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                        e.preventDefault();
                        const f = e.dataTransfer.files[0];
                        if (f) {
                            setFile(f);
                            setPreviewUrl(URL.createObjectURL(f));
                            setResult(null);
                        }
                    }}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        onChange={handleFileChange}
                        accept="image/*"
                    />

                    {previewUrl ? (
                        <div className="space-y-4">
                            <img
                                src={previewUrl}
                                alt="Preview"
                                className="max-h-64 mx-auto rounded-lg shadow-md object-contain"
                            />
                            <div className="flex items-center justify-center gap-4">
                                <span className="text-gray-600 font-medium">{file?.name}</span>
                                <button
                                    onClick={(e) => { e.stopPropagation(); handleClear(); }}
                                    className="text-red-500 hover:text-red-600 text-sm font-medium"
                                >
                                    Remove
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            <Upload className="w-12 h-12 mx-auto text-gray-400" />
                            <p className="text-gray-600 font-medium">Drop a blur wagon frame here</p>
                            <p className="text-gray-400 text-sm">or click to browse</p>
                        </div>
                    )}
                </div>

                <button
                    onClick={handleCompare}
                    disabled={!file || loading}
                    className={`w-full mt-4 py-4 rounded-lg font-bold text-white shadow-lg flex items-center justify-center gap-2 transition-all
                        ${!file || loading
                            ? 'bg-gray-300 cursor-not-allowed'
                            : 'bg-primary hover:bg-primary-light active:scale-[0.99]'}`}
                >
                    {loading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>{loadingStatus || 'Processing...'}</span>
                        </>
                    ) : (
                        <>
                            <span>Run Pipeline Comparison</span>
                            <ArrowRight className="w-5 h-5" />
                        </>
                    )}
                </button>
            </div>

            {/* Results Section */}
            {result && (
                <>
                    {/* Improvement Summary */}
                    {result.improvement && (
                        <div className="bg-white p-6 rounded-xl border border-gray-300 shadow-sm">
                            <h3 className="font-bold text-lg mb-4 text-primary flex items-center gap-2">
                                <CheckCircle className="w-5 h-5 text-success" />
                                Improvement Summary
                            </h3>
                            <div className="grid grid-cols-3 gap-4">
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <p className="text-gray-500 text-sm mb-1">Doors Detected</p>
                                    <div className="flex items-end gap-2">
                                        <span className="text-3xl font-bold text-gray-700">
                                            {result.improvement.doors_detected.blur}
                                        </span>
                                        <span className="text-gray-400 mb-1">→</span>
                                        <span className="text-3xl font-bold text-accent">
                                            {result.improvement.doors_detected.deblur}
                                        </span>
                                        {result.improvement.doors_detected.improvement > 0 && (
                                            <span className="text-success text-sm font-medium bg-success/10 px-1.5 py-0.5 rounded">
                                                +{result.improvement.doors_detected.improvement}
                                            </span>
                                        )}
                                    </div>
                                </div>
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <p className="text-gray-500 text-sm mb-1">Damage Detected</p>
                                    <div className="flex items-end gap-2">
                                        <span className="text-3xl font-bold text-gray-700">
                                            {result.improvement.damage_detected?.blur || 0}
                                        </span>
                                        <span className="text-gray-400 mb-1">→</span>
                                        <span className="text-3xl font-bold text-accent">
                                            {result.improvement.damage_detected?.deblur || 0}
                                        </span>
                                    </div>
                                </div>
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <p className="text-gray-500 text-sm mb-1">OCR Text Detected</p>
                                    <div className="flex items-end gap-2">
                                        <span className="text-3xl font-bold text-gray-700">
                                            {result.improvement.ocr_detected.blur}
                                        </span>
                                        <span className="text-gray-400 mb-1">→</span>
                                        <span className="text-3xl font-bold text-accent">
                                            {result.improvement.ocr_detected.deblur}
                                        </span>
                                        {result.improvement.ocr_detected.improvement > 0 && (
                                            <span className="text-success text-sm font-medium bg-success/10 px-1.5 py-0.5 rounded">
                                                +{result.improvement.ocr_detected.improvement}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Side by Side Comparison */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <ResultCard
                            title="Blur Frame (Original)"
                            data={result.blur_results}
                            isDeblur={false}
                        />
                        <ResultCard
                            title="Deblurred Frame (Enhanced)"
                            data={result.deblur_results}
                            isDeblur={true}
                        />
                    </div>
                </>
            )}
        </div>
    );
};

export default ComparePage;
