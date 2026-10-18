// Only used by the GitHub Pages build (tools/build_static_site.py bundles this file in; the
// Flask-served app never loads it, so it has no effect when you run the app normally).
//
// GitHub Pages serves static files only - it cannot run this project's Flask/Python backend - so
// a GitHub Pages deploy is the frontend alone, talking to a separately hosted copy of the backend
// (for example the Render deployment described in the README). Point this at that backend's URL,
// with no trailing slash:
window.KAGGLE_API_BASE = 'https://your-backend-url.onrender.com';

// For sign-in/save-to-account to work from GitHub Pages too (not just anonymous search/analyze/
// export), the backend also needs to be told to trust this exact page's origin - see
// DATASCOPIA_TRUSTED_ORIGIN in .env.example / render.yaml. Without that, the app still works for
// everything that doesn't require being signed in.
