/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        sideBar: "#0A284B",
        sidebarSelected: "#57e1ff",
      },
    }
  },
  plugins: [],
}
