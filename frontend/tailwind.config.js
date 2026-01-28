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
                    DEFAULT: '#2B2E6D',
                    light: '#4C4FB3',
                    dark: '#1E1F4B',
                },
                accent: {
                    DEFAULT: '#7C6CF2',
                    hover: '#6A5AE0',
                    light: '#C7C1FF',
                },
                success: '#22c55e',
                danger: '#ef4444',
                warning: '#f59e0b',
            },
        },
    },
    plugins: [],
}
