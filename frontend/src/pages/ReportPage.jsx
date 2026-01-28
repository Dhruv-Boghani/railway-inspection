import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Download, Eye, Loader, Calendar, Camera, Train, CheckCircle, Clock, Video } from 'lucide-react';
import axios from 'axios';
import NoJobFound from '../components/NoJobFound';

const ReportPage = () => {
    const navigate = useNavigate();
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchJobs = async () => {
            try {
                const response = await axios.get('http://localhost:8000/jobs');
                // Filter only completed jobs and sort by ID descending (latest first)
                const completedJobs = response.data
                    .filter(job => job.status === 'completed')
                    .sort((a, b) => b.id - a.id);
                setJobs(completedJobs);
            } catch (error) {
                console.error('Error fetching jobs:', error);
            } finally {
                setLoading(false);
            }
        };
        fetchJobs();
    }, []);

    const handleViewReport = (jobId) => {
        navigate(`/report/${jobId}`);
    };

    const formatDate = (timestamp) => {
        if (!timestamp) return null;
        return new Date(timestamp).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
        });
    };

    const formatTime = (timestamp) => {
        if (!timestamp) return null;
        return new Date(timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getRelativeTime = (timestamp) => {
        if (!timestamp) return 'Recently';
        const now = new Date();
        const date = new Date(timestamp);
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} min ago`;
        if (diffHours < 24) return `${diffHours} hours ago`;
        if (diffDays === 1) return 'Yesterday';
        if (diffDays < 7) return `${diffDays} days ago`;
        return formatDate(timestamp);
    };

    const getCameraLabel = (job) => {
        const angle = job.camera_angle || job.direction || 'SIDE';
        if (angle === 'TOP') return 'Top View';
        if (angle === 'LR') return 'Side: Left → Right';
        if (angle === 'RL') return 'Side: Right → Left';
        return 'Side View';
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="text-center">
                    <Loader className="w-10 h-10 animate-spin text-primary mx-auto mb-3" />
                    <p className="text-gray-600 font-medium">Loading inspection reports...</p>
                </div>
            </div>
        );
    }

    if (jobs.length === 0) {
        return <NoJobFound message="No Reports Found" />;
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-start space-x-4">
                    <div className="bg-primary text-white p-3 rounded-lg">
                        <FileText className="w-8 h-8" />
                    </div>
                    <div className="flex-1">
                        <h2 className="text-2xl font-bold text-primary">Inspection Reports</h2>
                        <p className="text-gray-500 mt-1">
                            View and download comprehensive reports of all completed inspections
                        </p>
                    </div>
                    <div className="text-right">
                        <div className="text-3xl font-bold text-primary">{jobs.length}</div>
                        <div className="text-sm text-gray-500">Total Reports</div>
                    </div>
                </div>
            </div>

            {/* Reports Grid */}
            {jobs.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                    {jobs.map((job) => (
                        <div
                            key={job.id}
                            className="bg-white rounded-xl shadow-sm border border-gray-300 overflow-hidden hover:shadow-lg transition-all hover:border-primary group"
                        >
                            {/* Card Header */}
                            <div className="bg-white p-4 border-b border-gray-200">
                                <div className="flex justify-between items-start">
                                    <div className="flex items-center space-x-3">
                                        <div className="bg-primary/10 p-2.5 rounded-lg">
                                            <Train className="w-5 h-5 text-primary" />
                                        </div>
                                        <div>
                                            <div className="text-xs text-gray-500 font-medium">Inspection Report</div>
                                            <div className="text-xl font-bold text-primary">Job #{job.id}</div>
                                        </div>
                                    </div>
                                    <div className="flex items-center space-x-1 bg-success/10 text-success px-2.5 py-1 rounded-full text-xs font-semibold border border-success/30">
                                        <CheckCircle className="w-3 h-3" />
                                        <span>Completed</span>
                                    </div>
                                </div>
                            </div>

                            {/* Card Body */}
                            <div className="p-4 space-y-4">
                                {/* Info Grid */}
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-gray-50 rounded-lg p-3">
                                        <div className="flex items-center space-x-2 text-gray-400 text-xs mb-1">
                                            <Calendar className="w-3.5 h-3.5" />
                                            <span>Date</span>
                                        </div>
                                        <div className="font-semibold text-gray-700">
                                            {formatDate(job.created_at) || getRelativeTime(job.created_at)}
                                        </div>
                                    </div>
                                    <div className="bg-gray-50 rounded-lg p-3">
                                        <div className="flex items-center space-x-2 text-gray-400 text-xs mb-1">
                                            <Clock className="w-3.5 h-3.5" />
                                            <span>Time</span>
                                        </div>
                                        <div className="font-semibold text-gray-700">
                                            {formatTime(job.created_at) || 'N/A'}
                                        </div>
                                    </div>
                                    <div className="bg-gray-50 rounded-lg p-3">
                                        <div className="flex items-center space-x-2 text-gray-400 text-xs mb-1">
                                            <Camera className="w-3.5 h-3.5" />
                                            <span>Camera</span>
                                        </div>
                                        <div className="font-semibold text-gray-700 text-sm">
                                            {getCameraLabel(job)}
                                        </div>
                                    </div>
                                    <div className="bg-gray-50 rounded-lg p-3">
                                        <div className="flex items-center space-x-2 text-gray-400 text-xs mb-1">
                                            <Train className="w-3.5 h-3.5" />
                                            <span>Wagons</span>
                                        </div>
                                        <div className="font-semibold text-gray-700">
                                            {job.total_wagons || 'N/A'}
                                        </div>
                                    </div>
                                </div>

                                {/* Video File */}
                                {job.filename && (
                                    <div className="flex items-center space-x-2 text-sm text-gray-500 bg-gray-50 rounded-lg px-3 py-2">
                                        <Video className="w-4 h-4 text-gray-400" />
                                        <span className="truncate flex-1" title={job.filename}>
                                            {job.filename}
                                        </span>
                                    </div>
                                )}

                                {/* Action Buttons */}
                                <div className="flex space-x-2 pt-2">
                                    <button
                                        onClick={() => handleViewReport(job.id)}
                                        className="flex-1 flex items-center justify-center space-x-2 bg-primary text-white py-2.5 rounded-lg hover:bg-primary-light transition-colors font-medium"
                                    >
                                        <Eye className="w-4 h-4" />
                                        <span>View Report</span>
                                    </button>
                                    <button
                                        onClick={() => handleViewReport(job.id)}
                                        className="flex items-center justify-center space-x-2 bg-primary text-white px-4 py-2.5 rounded-lg hover:bg-accent-hover transition-colors font-medium"
                                    >
                                        <Download className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-12 text-center">
                    <div className="bg-gray-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                        <FileText className="w-10 h-10 text-gray-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-700 mb-2">No Reports Available</h3>
                    <p className="text-gray-500 mb-6">
                        Upload and process a video to generate your first inspection report.
                    </p>
                    <button
                        onClick={() => navigate('/upload')}
                        className="bg-primary text-white px-6 py-3 rounded-lg hover:bg-primary-light transition-colors font-medium"
                    >
                        Upload Video
                    </button>
                </div>
            )}
        </div>
    );
};

export default ReportPage;
