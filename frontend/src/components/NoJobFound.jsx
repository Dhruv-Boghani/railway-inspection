import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileVideo } from 'lucide-react';

const NoJobFound = ({ message = "Upload Video First" }) => {
    const navigate = useNavigate();

    return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] text-center p-8">
            <div className="bg-blue-50 p-6 rounded-full mb-6">
                <FileVideo className="w-16 h-16 text-blue-500" />
            </div>

            <h2 className="text-2xl font-bold text-gray-800 mb-3">
                {message}
            </h2>

            <p className="text-gray-500 max-w-md mb-8">
                No inspection data found. Please upload a new wagon inspection video to start the analysis process.
            </p>

            <button
                onClick={() => navigate('/upload')}
                className="flex items-center space-x-2 bg-primary text-white px-8 py-3 rounded-xl hover:bg-primary/90 transition-colors duration-200 font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
            >
                <Upload className="w-5 h-5" />
                <span>Go to Upload Page</span>
            </button>
        </div>
    );
};

export default NoJobFound;
