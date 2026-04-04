/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          400: '#818cf8',
          500: '#6366f1',
          600: '#4f46e5',
        },
        surface: {
          0: '#020617',
          1: '#0f172a',
          2: '#1e293b',
          3: '#334155',
        }
      }
    },
  },
  plugins: [],
}
