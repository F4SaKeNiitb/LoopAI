module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
  corePlugins: {
    preflight: true,
  },
  plugins: [
    function ({ addComponents }) {
      addComponents({
        '.btn-primary': {
          '@apply px-6 py-3 rounded-lg text-white font-medium shadow-md transform transition duration-200 hover:scale-105 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500': {}
        },
        '.btn-primary:disabled': {
          '@apply opacity-70 cursor-not-allowed': {}
        },
        '.btn-gradient': {
          '@apply bg-gradient-to-r from-blue-600 to-indigo-700 text-white rounded-lg font-medium shadow-md hover:from-blue-700 hover:to-indigo-800 transition duration-200': {}
        },
        '.focus-ring': {
          '@apply focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50': {}
        }
      })
    }
  ]
};