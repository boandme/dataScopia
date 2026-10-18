// Firebase Authentication in the browser (ES module, loaded after app.js and social.js).
//
// Firebase handles identity - email + password, Google sign-in, password reset, email verification. After every
// sign-in this module sends the user's ID token to the server (POST /api/auth/firebase), which verifies it against
// Google's public keys and starts a normal session cookie. All app data then flows through that session; the
// browser never talks to the Realtime Database or Cloud Storage directly (the bundled security rules deny it).

import {initializeApp} from 'https://www.gstatic.com/firebasejs/12.19.0/firebase-app.js';
import {
  getAuth, onAuthStateChanged, createUserWithEmailAndPassword, signInWithEmailAndPassword, signInWithPopup, GoogleAuthProvider,
  sendPasswordResetEmail, sendEmailVerification, updateProfile, updatePassword, reauthenticateWithCredential, reauthenticateWithPopup,
  EmailAuthProvider, signOut, deleteUser,
} from 'https://www.gstatic.com/firebasejs/12.19.0/firebase-auth.js';

const API_BASE = window.KAGGLE_API_BASE || new URLSearchParams(window.location.search).get('api') || (window.location.protocol === 'file:' ? 'http://localhost:8000' : '');

const FRIENDLY = {
  'auth/invalid-credential': 'Incorrect email or password.',
  'auth/wrong-password': 'Incorrect email or password.',
  'auth/user-not-found': 'Incorrect email or password.',
  'auth/invalid-email': 'Enter a valid email address.',
  'auth/email-already-in-use': 'An account with that email already exists. Try signing in.',
  'auth/weak-password': 'Choose a stronger password (at least 8 characters).',
  'auth/too-many-requests': 'Too many attempts. Please wait a few minutes and try again.',
  'auth/network-request-failed': 'Network problem - check your connection and try again.',
  'auth/popup-closed-by-user': 'The Google sign-in window was closed before finishing.',
  'auth/popup-blocked': 'Your browser blocked the Google sign-in window. Allow pop-ups and try again.',
  'auth/cancelled-popup-request': 'Another sign-in window is already open.',
  'auth/operation-not-allowed': 'This sign-in method is not enabled yet in the Firebase console (Authentication > Sign-in method).',
  'auth/unauthorized-domain': 'This site\'s domain is not authorized in the Firebase console (Authentication > Settings > Authorized domains).',
  'auth/requires-recent-login': 'For your security, please sign in again and retry.',
  'auth/user-disabled': 'This account has been disabled.',
};

function friendly(error) {
  const message = FRIENDLY[error?.code];
  const wrapped = new Error(message || error?.message || 'Something went wrong. Please try again.');
  wrapped.code = error?.code;
  return wrapped;
}

async function serverPost(path, body) {
  // 'include' so the session cookie still flows for a cross-origin GitHub Pages + trusted-origin
  // API deployment; equivalent to 'same-origin' for the default same-origin deployment.
  const response = await fetch(`${API_BASE}${path}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, credentials: 'include', body: JSON.stringify(body)});
  let data = null;
  try { data = await response.json(); } catch (error) { data = null; }
  if (!response.ok) throw new Error((data && data.detail) || `Request failed (${response.status}).`);
  return data;
}

async function boot() {
  const configResponse = await fetch(`${API_BASE}/api/auth/config`, {credentials: 'include'});
  const config = await configResponse.json();
  if (config.mode !== 'firebase' || !config.firebase) return null;

  const app = initializeApp(config.firebase);
  const auth = getAuth(app);
  const google = new GoogleAuthProvider();

  async function exchange(user, displayName = '') {
    const idToken = await user.getIdToken(true);
    const {user: serverUser} = await serverPost('/api/auth/firebase', {id_token: idToken, display_name: displayName});
    return serverUser;
  }

  const hasPasswordProvider = () => Boolean(auth.currentUser?.providerData.some((p) => p.providerId === 'password'));

  async function reauthenticate(password) {
    const user = auth.currentUser;
    if (!user) throw new Error('Please sign in again.');
    if (hasPasswordProvider()) {
      if (!password) throw new Error('Enter your password to confirm.');
      await reauthenticateWithCredential(user, EmailAuthProvider.credential(user.email, password));
    } else {
      await reauthenticateWithPopup(user, google);
    }
  }

  return {
    mode: 'firebase',
    hasPassword: hasPasswordProvider,
    async signUp(email, password, name) {
      try {
        const credential = await createUserWithEmailAndPassword(auth, email, password);
        if (name) await updateProfile(credential.user, {displayName: name});
        sendEmailVerification(credential.user).catch(() => { /* optional; never blocks sign-up */ });
        return await exchange(credential.user, name);
      } catch (error) { throw error.code ? friendly(error) : error; }
    },
    async signIn(email, password) {
      try {
        const credential = await signInWithEmailAndPassword(auth, email, password);
        return await exchange(credential.user);
      } catch (error) { throw error.code ? friendly(error) : error; }
    },
    async signInWithGoogle() {
      try {
        const credential = await signInWithPopup(auth, google);
        return await exchange(credential.user);
      } catch (error) { throw error.code ? friendly(error) : error; }
    },
    async resetPassword(email) {
      try { await sendPasswordResetEmail(auth, email); } catch (error) { throw friendly(error); }
    },
    // If Firebase still remembers the user but our session cookie is gone (e.g. after a redeploy), sign in silently.
    restoreSession() {
      return new Promise((resolve) => {
        const stop = onAuthStateChanged(auth, async (user) => {
          stop();
          if (!user) { resolve(null); return; }
          try { resolve(await exchange(user)); } catch (error) { resolve(null); }
        });
      });
    },
    async signOut() { try { await signOut(auth); } catch (error) { /* already signed out */ } },
    async updateName(name) {
      if (!auth.currentUser) throw new Error('Please sign in again.');
      try {
        await updateProfile(auth.currentUser, {displayName: name});
        return await exchange(auth.currentUser, name);
      } catch (error) { throw error.code ? friendly(error) : error; }
    },
    async changePassword(current, next) {
      if (!hasPasswordProvider()) throw new Error('This account signs in with Google, so it has no password to change here.');
      try {
        await reauthenticate(current);
        await updatePassword(auth.currentUser, next);
      } catch (error) { throw error.code ? friendly(error) : error; }
    },
    async deleteAccount(password) {
      try {
        await reauthenticate(password);
        const idToken = await auth.currentUser.getIdToken(true);   // fresh: the server requires a recent sign-in
        await serverPost('/api/auth/delete', {id_token: idToken});
        await deleteUser(auth.currentUser);
      } catch (error) { throw error.code ? friendly(error) : error; }
    },
  };
}

boot().then((api) => window.__resolveFirebase && window.__resolveFirebase(api)).catch((error) => {
  console.error('Firebase could not start:', error);
  if (window.__resolveFirebase) window.__resolveFirebase(null);
});
