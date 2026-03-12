/** @type {import('tailwindcss').Config} */
module.exports = {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {},
	},
	plugins: [require("@tailwindcss/typography"),require("daisyui")],
	daisyui: {
		// ← Change these two values to any DaisyUI theme names you want
		themes: ["dracula", "light"], // [dark theme, light theme]
		logs: false,
	  }
}
