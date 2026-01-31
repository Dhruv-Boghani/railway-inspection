import React from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useJob } from "../context/JobContext";
import {
    Upload,
    Loader,
    BarChart3,
    FileText,
    Layers,
    Train,
    Split,
    ChevronLeft,
    ChevronRight
} from "lucide-react";

const Sidebar = ({ isCollapsed, setIsCollapsed }) => {
    const location = useLocation();
    const { currentJobId, isProcessing } = useJob();

    const navItems = [
        { path: "/upload", icon: Upload, label: "Upload Video" },
        { path: currentJobId ? `/processing/${currentJobId}` : "#", icon: Loader, label: "Processing", disabled: !currentJobId, animate: isProcessing },
        { path: currentJobId ? `/results/${currentJobId}` : "#", icon: BarChart3, label: "Results", disabled: !currentJobId },
        { path: currentJobId ? `/analysis/${currentJobId}` : "#", icon: Layers, label: "Detailed Analysis", disabled: !currentJobId },
        { path: "/reports", icon: FileText, label: "Reports" },
        // { path: "/compare", icon: Split, label: "Compare Frames" },
    ];

    const isActive = (path) => {
        if (path === "#") return false;
        return location.pathname.startsWith(path.split('/')[1] ? `/${path.split('/')[1]}` : path);
    };

    return (
        <div
            className={`fixed left-0 top-0 h-screen bg-sidebar-gradient text-slate-300 flex flex-col z-50 transition-all duration-300 border-r border-white/10 shadow-2xl shadow-indigo-900/20 backdrop-blur-md ${isCollapsed ? 'w-20' : 'w-72'
                }`}
        >
            {/* Logo Header */}
            <div className={`p-6 ${isCollapsed ? 'flex justify-center px-4' : ''}`}>
                <div className={`flex items-center gap-3 ${isCollapsed ? 'justify-center' : ''}`}>
                    <div className="relative group">
                        <div className="absolute inset-0 bg-indigo-500 blur-lg opacity-40 group-hover:opacity-60 transition-opacity rounded-xl"></div>
                        <div className="relative bg-gradient-to-br from-indigo-500 to-indigo-700 p-2.5 rounded-xl flex-shrink-0 border border-white/10 shadow-lg">
                            <Train className="w-6 h-6 text-white" />
                        </div>
                    </div>
                    {!isCollapsed && (
                        <div>
                            <h1 className="font-bold text-lg leading-tight text-white tracking-wide">Railway Insp.</h1>
                            <p className="text-xs text-indigo-200 mt-0.5 font-medium tracking-wide opacity-80">Wagon Analysis Sys.</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-4 space-y-2 mt-4">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const active = isActive(item.path);

                    if (item.disabled) {
                        return (
                            <div
                                key={item.path}
                                className={`flex items-center gap-3 px-4 py-3 text-slate-600 cursor-not-allowed rounded-xl ${isCollapsed ? 'justify-center px-2' : ''
                                    }`}
                                title={isCollapsed ? item.label : ''}
                            >
                                <Icon className={`w-5 h-5 flex-shrink-0 ${item.animate ? 'animate-spin' : ''}`} />
                                {!isCollapsed && <span className="text-sm font-medium">{item.label}</span>}
                            </div>
                        );
                    }

                    return (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group relative overflow-hidden ${isCollapsed ? 'justify-center px-2' : ''
                                } ${active
                                    ? 'text-white shadow-[0_0_20px_-5px_rgba(99,102,241,0.4)]'
                                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                                }`}
                            title={isCollapsed ? item.label : ''}
                        >
                            {active && (
                                <div className="absolute inset-0 bg-white/10 backdrop-blur-md border border-white/10 rounded-xl" />
                            )}
                            <Icon className={`w-5 h-5 flex-shrink-0 z-10 transition-transform duration-300 ${active ? 'text-indigo-400' : ''} ${!active && 'group-hover:scale-110 group-hover:text-indigo-300'}`} />
                            {!isCollapsed && <span className={`text-sm font-medium z-10 ${active ? 'font-semibold tracking-wide' : ''}`}>{item.label}</span>}
                        </NavLink>
                    );
                })}
            </nav>

            {/* Collapse Toggle Button - Above Footer */}
            <div className="px-4 pb-4">
                <button
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    className="w-full flex items-center justify-center gap-3 py-3 rounded-xl text-slate-500 hover:bg-white/5 hover:text-indigo-300 transition-colors border border-transparent hover:border-white/5"
                    title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
                >
                    {isCollapsed ? (
                        <ChevronRight className="w-5 h-5" />
                    ) : (
                        <>
                            <ChevronLeft className="w-5 h-5" />
                            <span className="text-sm font-medium">Collapse Sidebar</span>
                        </>
                    )}
                </button>
            </div>

            {/* Footer */}
            {!isCollapsed && (
                <div className="px-6 py-6 text-xs text-slate-500 text-center border-t border-white/5">
                    <p className="font-semibold text-slate-400 mb-1">Developed by Team HackHustler</p>
                    <p className="opacity-60">© 2026 All Rights Reserved</p>
                </div>
            )}
        </div>
    );
};

export default Sidebar;
