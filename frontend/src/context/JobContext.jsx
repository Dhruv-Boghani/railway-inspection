import React, { createContext, useState, useContext, useEffect } from 'react';

const JobContext = createContext();

export const JobProvider = ({ children }) => {
    const [currentJobId, setCurrentJobId] = useState(localStorage.getItem('currentJobId') || null);
    const [isProcessing, setIsProcessing] = useState(false);

    // Update localStorage when state changes
    useEffect(() => {
        if (currentJobId) {
            localStorage.setItem('currentJobId', currentJobId);
        }
    }, [currentJobId]);

    return (
        <JobContext.Provider value={{ currentJobId, setCurrentJobId, isProcessing, setIsProcessing }}>
            {children}
        </JobContext.Provider>
    );
};

export const useJob = () => useContext(JobContext);
