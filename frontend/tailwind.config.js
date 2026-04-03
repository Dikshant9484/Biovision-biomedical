/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Clash Display', 'Plus Jakarta Sans', 'sans-serif'],
      },
      colors: {
        ink: {
          950: '#060A12',
          900: '#0D1421',
          800: '#141D2E',
          700: '#1C2840',
          600: '#243352',
        },
        neon: {
          cyan: '#00E5FF',
          teal: '#00C9A7',
          blue: '#4B9EFF',
          green: '#00F5A0',
        },
        danger: '#FF4D6D',
        safe:   '#00C9A7',
      },
      animation: {
        'scan': 'scan 2.5s ease-in-out infinite',
        'pulse-ring': 'pulse-ring 2s ease-out infinite',
        'fade-up': 'fade-up 0.5s ease forwards',
        'shimmer': 'shimmer 1.8s linear infinite',
      },
      keyframes: {
        scan: {
          '0%,100%': { transform: 'translateY(0%)', opacity: '0.6' },
          '50%':      { transform: 'translateY(100%)', opacity: '1' },
        },
        'pulse-ring': {
          '0%':   { transform: 'scale(0.9)', opacity: '0.8' },
          '100%': { transform: 'scale(1.6)', opacity: '0' },
        },
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-400px 0' },
          '100%': { backgroundPosition: '400px 0' },
        },
      },
    },
  },
  plugins: [],
}
