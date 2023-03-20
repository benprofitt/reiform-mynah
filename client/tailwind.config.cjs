/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        floating: "0px 0px 4px 1px #00000026",
      },
      colors: {
        sideBar: "#0A284B",
        sidebarSelected: "#57e1ff",
        grey: "#f9f9f9",
        grey: "#f9f9f9",
      },
    }
  },
  plugins: [],
}
