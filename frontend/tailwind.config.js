module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        floating: "0px 0px 4px 1px #00000026",
        huge: '0px 5px 10px rgba(175, 175, 175, 0.15), 0px 10px 20px rgba(175, 175, 175, 0.15)'
      },
      colors: {
        sideBar: "#0A284B",
        sidebarSelected: "#57e1ff",
        linkblue: "#0085FF",
        grey: "#f9f9f9",
        grey1: "#e9e9e9",
        grey2: "#6e757d",
        grey3: "#e8e8e8",
        clearGrey3: "rgba(252,252,252,0)",
        grey4: "#e8e9eb",
        grey5: "#828282",
        grey6: "#bbbec2",
      },
    },
  },
  plugins: [],
  important: true,
};
