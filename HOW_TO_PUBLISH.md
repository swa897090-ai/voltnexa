# Publish Voltnexa as a real website (no Colab, no running code)

Put your two data files in this folder first: `ev_charging_sessions.csv` and `stations.csv`
(next to `voltnexa_pro_v3.py`). Do not change any file names.

## Option A - Render.com (easiest, link like https://voltnexa.onrender.com)
1. Create a free account at https://github.com and make a new repository called `voltnexa`.
2. Upload everything in this folder to it (Add file > Upload files), including the two CSV files.
3. Create a free account at https://render.com (sign in with GitHub).
4. New + > Blueprint > choose your `voltnexa` repository > Apply. (Or New + > Web Service, build command
   `pip install -r requirements.txt`, start command from the `Procfile`.)
5. Wait 5-10 minutes. Render gives you a link. Open it in Chrome, Edge or on your phone: it opens the Register page.
Notes:
 * Free plan: the site sleeps after 15 minutes without visitors and takes about a minute to wake up.
 * Free plan: the disk is temporary, so registered accounts are erased whenever the site restarts or is redeployed.
   To keep accounts, add a Persistent Disk (paid plan) mounted at /var/data and set the environment variable
   VOLTNEXA_DATA=/var/data.

## Option B - PythonAnywhere.com (free, keeps registered accounts, link like https://USERNAME.pythonanywhere.com)
1. Create a free account at https://www.pythonanywhere.com
2. Files tab: create the folder `voltnexa` and upload all files from this folder (including the two CSV files).
3. Consoles tab > Bash console:
       mkvirtualenv voltnexa --python=python3.10
       pip install -r ~/voltnexa/requirements.txt
4. Web tab > Add a new web app > Manual configuration > Python 3.10.
   Virtualenv: /home/USERNAME/.virtualenvs/voltnexa
   WSGI file: replace its contents with `pythonanywhere_wsgi.py` (change USERNAME).
5. Click Reload. Open https://USERNAME.pythonanywhere.com
Notes: log in to PythonAnywhere and press the "Run until" button on the Web tab about every 3 months to keep the site
running. The free plan blocks some outside websites, so the road route may fall back to an estimate.

## Real OTP SMS
Without an SMS service the OTP is shown on the register page (demo). For real SMS, create a Twilio account and add the
environment variables TWILIO_SID, TWILIO_TOKEN and TWILIO_FROM (Render: Environment tab; PythonAnywhere: top of the WSGI file
using os.environ["TWILIO_SID"] = "..."). Then set VOLTNEXA_DEMO_OTP=0 so the OTP is no longer displayed.

## Phone app
Open your website link in Chrome on the phone > menu > Install app / Add to Home screen. Hosting gives it https,
so it installs as a full app.
