# Deploying the live lab so your professor can open it himself

You've decided: this runs with **your** OpenAI key, server-side, for anyone
who has the link, with **no passcode and no spending cap**. That's a real
decision with real exposure — anyone who gets the URL can spend your OpenAI
balance for as long as it's live. Two things follow from that:

1. **Treat the URL like a secret.** Send it directly to your professor, don't
   post it anywhere public or leave it in a shareable doc.
2. **Take the app down (or rotate the key) once the demo is over.** A free
   host that "sleeps" when idle (see below) gives you some natural protection
   between uses, but it is not a substitute for actually turning it off.

The rest of this doc gets it from your laptop to a public URL.

## What changed to make this possible

- `app.py` now calls `_auto_configure_from_env()` once at startup: if
  `OPENAI_API_KEY` is already set in the environment, it connects
  automatically and the page hides the "paste your key" form entirely. Locally
  without that env var, step 1 still works exactly as before, for your own
  testing.
- `Procfile` and `gunicorn` (added to `requirements.txt`) replace Flask's
  built-in dev server, which prints its own warning against exactly this use
  ("do not use in a production deployment").
- The app keeps its state (loaded dataset, optimized program) in plain Python
  memory, not a database - so it **must** run as a single process
  (`--workers 1` in the Procfile). Multiple threads inside that one process
  (`--threads 8`) still let it handle several visitors' clicks at once; do not
  raise `--workers` above 1 without also adding a real datastore.

## Step 1 — put the project in a git repo and push it to GitHub

Render (and most similar hosts) deploy by connecting to a GitHub repo. This
project isn't a git repo yet. From `C:\Users\hp\dspy-demo`:

```powershell
git init
git add app.py common.py templates static requirements.txt Procfile .gitignore README.md README_WEBAPP.md CODE_WALKTHROUGH.md *.py
git commit -m "DSPy live lab"
```

Then create a new repository on github.com (private is fine), and follow
GitHub's instructions to push an existing local repo — it'll give you exact
`git remote add origin ...` and `git push` commands with your own repo URL.
**Do not commit your API key anywhere in this repo** - it only ever goes into
Render's environment-variable dashboard (step 3), never into a file.

## Step 2 — create a Render account and a new Web Service

[render.com](https://render.com) currently offers a free web-service tier
without requiring a card for a basic app like this one — but hosting free
tiers change their terms fairly often, so double-check current pricing before
you commit. (Railway and Fly.io are reasonable alternatives if Render's terms
have changed; **avoid PythonAnywhere's free tier** specifically - it
restricts outbound network calls to an allowlist that most likely does not
include `api.openai.com`, which would break every real call this app makes.)

1. Sign up / log in, then **New +** → **Web Service**.
2. Connect the GitHub repo you just pushed.
3. Settings:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --worker-class gthread --workers 1 --threads 8 --timeout 120 app:app`
     (Render may auto-detect this from the `Procfile` - either way, make sure it matches.)
   - **Instance Type**: Free

## Step 3 — add your API key as an environment variable, on Render's site

In the Web Service's **Environment** tab, add:

| Key | Value |
| --- | --- |
| `OPENAI_API_KEY` | your real key |

Optionally also add `DSPY_DEMO_MODEL` if you want a model other than
`openai/gpt-5.4-nano`. This is the *only* place the real key should ever be
typed — not into this chat, not into a file in the repo.

## Step 4 — deploy, warm it up, share the link

Click **Deploy**. Render builds and starts the app and gives you a URL like
`https://your-app-name.onrender.com`.

- **Free-tier apps sleep after inactivity** and take roughly 30-60 seconds to
  wake up on the next request. Open the link yourself a few minutes before
  your professor does, so it's already warm.
- Because state lives in memory, a restart (sleeping, or a new deploy) resets
  any loaded dataset or optimized program back to nothing - the page still
  works fine, it just starts step 2 over. That's a fine, honest tradeoff for
  a demo tool; it would need a real database to survive restarts, which is
  more than this needs.
- Send the URL to your professor. He opens it, sees step 1 already says
  "Connected," and goes straight to loading a dataset.

## If something looks wrong after deploying

- **Every real call fails with the same error** → check the exact
  `OPENAI_API_KEY` value in Render's dashboard for stray whitespace, and that
  the model name in `DSPY_DEMO_MODEL` (if set) is one your account actually
  has access to.
- **The page loads but step 1 still asks for a key** → the environment
  variable isn't visible to the running process; re-check the spelling
  `OPENAI_API_KEY` and redeploy.
- **It worked, then suddenly nothing loads** → likely the free-tier app went
  to sleep; give it a minute on the first request back.
