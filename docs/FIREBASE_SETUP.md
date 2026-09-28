# Firebase setup (accounts + storage)

DataScopia uses your Firebase project (`datascopia`) in two independent ways:

| What | Needs | Without it |
|---|---|---|
| **Accounts** - Firebase Authentication (email + password, Google, password reset, email verification) | Steps 1 and 2 below. No secret is required. | Set `DATASCOPIA_AUTH=local` for built-in email + password accounts. |
| **Storage** - projects, published analyses and the dataset catalogue in the **Realtime Database**; data files in Cloud Storage | Steps 3 to 5 (a service-account key + the database URL). | Data stays in a local SQLite file + `data/` folder, which is **wiped on every deploy on free hosts**. |

The web config you pasted (apiKey, appId, ...) is public by design and is already built in. The Firebase Analytics
snippet is not used (the app does not track users). Firestore is **not** used anywhere - documents live in the
Realtime Database instead.

## 1. Turn on sign-in methods
Firebase console -> **Build -> Authentication -> Get started -> Sign-in method**:
- enable **Email/Password**
- enable **Google** (choose a support email)

## 2. Authorize your domains
**Authentication -> Settings -> Authorized domains**: `localhost` is there already; add the domain you deploy to
(for example `datascopia.onrender.com`). Google sign-in and email links fail on unlisted domains.

## 3. Create the Realtime Database and a Storage bucket
- **Build -> Realtime Database -> Create database** -> start in *locked mode* (any region; the rules below replace
  the default ones). Once created, copy the URL shown at the top of the **Data** tab, e.g.
  `https://datascopia-default-rtdb.firebaseio.com` - you'll need it in step 5.
- **Build -> Storage -> Get started** (Firebase may ask you to upgrade to the Blaze pay-as-you-go plan for a
  bucket; usage at this scale is normally within the free quota, and you can set a budget alert).

## 4. Lock the database and bucket to the server
Deploy the rules in this repo, which deny every browser request (the server uses a service account, which is
authenticated as an admin and bypasses these rules, the same as the Admin SDK):

```
npm i -g firebase-tools
firebase login
firebase use datascopia
firebase deploy --only database,storage
```

(or paste `database.rules.json` and `storage.rules` into the console's Rules tabs for Realtime Database and Storage).

## 5. Give the server a service account
**Project settings -> Service accounts -> Generate new private key** downloads a JSON file. **Never commit it.**
Provide it to the server as environment variables, either the key JSON itself or its base64, plus the database URL
from step 3:

```
# Render / any host: paste the file's contents into a secret env var
FIREBASE_SERVICE_ACCOUNT={"type":"service_account", ...}
FIREBASE_DATABASE_URL=https://datascopia-default-rtdb.firebaseio.com

# local development: point at the file instead (keep it outside the repo)
FIREBASE_SERVICE_ACCOUNT_FILE=C:\keys\datascopia-service-account.json
FIREBASE_DATABASE_URL=https://datascopia-default-rtdb.firebaseio.com
```

When the key and database URL are present, the app switches to the Realtime Database + Cloud Storage automatically
(`DATASCOPIA_DATA=auto`). Check with `GET /api/auth/config` -> `"data_backend": "firebase"`. If the URL is missing
or wrong, the server logs a warning and falls back to local storage rather than failing to start.

## Environment variables
| Variable | Default | Purpose |
|---|---|---|
| `DATASCOPIA_AUTH` | `firebase` | `local` for built-in accounts |
| `DATASCOPIA_DATA` | `auto` | `firebase`, `local` or `auto` (Firebase when a key + database URL exist) |
| `FIREBASE_DATABASE_URL` | - | your Realtime Database URL (required for the Firebase storage backend) |
| `FIREBASE_SERVICE_ACCOUNT` / `_FILE` / `GOOGLE_APPLICATION_CREDENTIALS` | - | server credentials |
| `FIREBASE_PROJECT_ID` | `datascopia` | your project |
| `FIREBASE_STORAGE_BUCKET` | `<project>.firebasestorage.app` | Cloud Storage bucket |
| `FIREBASE_WEB_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `FIREBASE_APP_ID`, `FIREBASE_MESSAGING_SENDER_ID` | your values | public web config sent to the browser |

## How it fits together
1. The browser signs in with the Firebase SDK and gets an ID token.
2. It posts the token to `/api/auth/firebase`; the server verifies Google's signature, expiry, issuer and that the
   token was issued for *this* project, then starts a session cookie. No password reaches the server.
3. Projects, published analyses and the catalogue are Realtime Database records keyed by the user's Firebase `uid`;
   datasets are Parquet objects in Cloud Storage. Only the server touches them.
4. If the server restarts and loses sessions, the browser (still signed in to Firebase) silently starts a new one.

## What is *not* covered
- Each record is capped (`docstore.MAX_DOC_BYTES`, comfortably under 1 MB), so a project's findings/notes are
  capped too; the app rejects larger ones with a clear error before any network call.
- Stored-dataset size is capped at 80 MB per project (`DATASCOPIA_MAX_PROJECT_MB`).
- Live Firebase behaviour has only been tested against an in-memory fake of the REST APIs. After your first deploy,
  sign up, save a project, publish it and re-verify it once to confirm the console setup above.
