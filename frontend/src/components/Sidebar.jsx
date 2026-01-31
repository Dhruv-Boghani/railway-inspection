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
    ChevronRight,
    BookOpen
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
        { path: "/documentation", icon: BookOpen, label: "Documentation" },
    ];

    const isActive = (path) => {
        if (path === "#") return false;
        return location.pathname.startsWith(path.split('/')[1] ? `/${path.split('/')[1]}` : path);
    };

    return (
        <div
            className={`fixed left-0 top-0 h-screen bg-primary-dark text-white flex flex-col z-50 ${isCollapsed ? 'w-16' : 'w-56'
                }`}
        >
            {/* Logo Header */}
            <div className={`p-4 pt-12 pb-16 ${isCollapsed ? 'flex justify-center' : ''}`}>
                <div className={`flex items-center gap-3 ${isCollapsed ? 'justify-center' : ''}`}>
                    <div className="bg-accent p-2.5 rounded-lg flex-shrink-0">
                        <Train className="w-7 h-7 text-white" />
                    </div>
                    {!isCollapsed && (
                        <div>
                            <h1 className="font-semibold text-base leading-tight text-white">Railway Inspector</h1>
                            <p className="text-xs text-gray-400 mt-0.5">Wagon Analysis System</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-3 space-y-1">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const active = isActive(item.path);

                    if (item.disabled) {
                        return (
                            <div
                                key={item.path}
                                className={`flex items-center gap-3 px-3 py-3 text-gray-500 cursor-not-allowed rounded-lg ${isCollapsed ? 'justify-center' : ''
                                    }`}
                                title={isCollapsed ? item.label : ''}
                            >
                                <Icon className={`w-5 h-5 flex-shrink-0 ${item.animate ? 'animate-spin' : ''}`} />
                                {!isCollapsed && <span className="text-sm">{item.label}</span>}
                            </div>
                        );
                    }

                    return (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            className={`flex items-center gap-3 px-3 py-3 rounded-lg ${isCollapsed ? 'justify-center' : ''
                                } ${active
                                    ? 'bg-accent text-white font-medium'
                                    : 'text-gray-300 hover:bg-white/5 hover:text-white'
                                }`}
                            title={isCollapsed ? item.label : ''}
                        >
                            <Icon className={`w-5 h-5 flex-shrink-0 ${item.animate ? 'animate-spin' : ''}`} />
                            {!isCollapsed && <span className="text-sm">{item.label}</span>}
                        </NavLink>
                    );
                })}
            </nav>

            {/* Collapse Toggle Button - Above Footer */}
            <div className="px-3 pb-2">
                <button
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    className="w-full flex items-center justify-center gap-3 py-2.5 rounded-lg text-gray-400 hover:bg-white/5 hover:text-white"
                    title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
                >
                    {isCollapsed ? (
                        <ChevronRight className="w-6 h-6" />
                    ) : (
                        <>
                            <ChevronLeft className="w-6 h-6" />
                        </>
                    )}
                </button>
            </div>

            {/* Footer */}
            {!isCollapsed && (
                <div className="p-4 text-sm text-gray-500 text-center border-t border-white/10">
                    <p className="text-gray-400">Team</p>
                    <p>Missing Sem;colon</p>
                    <p>© 2026 All Rights Reserved</p>
                </div>
            )}
        </div>
    );
};

export default Sidebar;
