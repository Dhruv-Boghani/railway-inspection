import React, { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";

const Layout = () => {
    const location = useLocation();
    const isAnalysisPage = location.pathname.includes('/analysis');
    const [isCollapsed, setIsCollapsed] = useState(false);

    return (
        <div className="min-h-screen bg-gray-100">
            <Sidebar isCollapsed={isCollapsed} setIsCollapsed={setIsCollapsed} />
            <main
                className={`${isCollapsed ? 'ml-16' : 'ml-56'} ${isAnalysisPage ? 'p-0' : 'p-6'}`}
            >
                <Outlet />
            </main>
        </div>
    );
};

export default Layout;
