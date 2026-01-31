/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    DEFAULT: '#6366f1', // INDIGO-500 (Vibrant)
                    light: '#818cf8',   // INDIGO-400
                    dark: '#4f46e5',    // INDIGO-600
                    glow: '#4338ca',    // INDIGO-700
                },
                accent: {
                    DEFAULT: '#a855f7', // PURPLE-500
                    hover: '#9333ea',   // PURPLE-600
                    light: '#d8b4fe',   // PURPLE-300
                    glow: '#7e22ce',
                },
                slate: {
                    50: '#f8fafc',
                    100: '#f1f5f9',
                    200: '#e2e8f0',
                    300: '#cbd5e1',
                    400: '#94a3b8',
                    500: '#64748b',
                    600: '#475569',
                    700: '#334155',
                    800: '#1e293b',
                    900: '#0f172a',
                    950: '#020617', // DEEP BACKGROUND
                },
                success: '#10b981', // EMERALD-500
                danger: '#ef4444',  // RED-500
                warning: '#f59e0b', // AMBER-500
            },
            backgroundImage: {
                'sidebar-gradient': 'linear-gradient(to bottom, #0f172a, #1e1b4b, #0f172a)',
                'glass-gradient': 'linear-gradient(145deg, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.4))',
            },
        },
    },
    plugins: [],
}
