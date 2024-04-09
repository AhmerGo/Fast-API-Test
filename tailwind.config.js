module.exports = {
  purge: ["./src/**/*.{js,jsx,ts,tsx}", "./public/index.html"],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      backgroundImage: {
        hsu: "url('assets/HSU.png')",
      },
    },
    colors: {
      purple: "#581483",
      gold: "#FFC72C",
      white: "#FFFFFF",
      gray: {
        800: "#1F2937",
      },
      indigo: {
        600: "#4F46E5",
        700: "#4338CA",
        900: "#312E81",
      },
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
};
