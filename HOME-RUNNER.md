# Home runner (needed for YouTube streams)

YouTube refuses to serve video to data-center servers like GitHub's ("Sign in to confirm you're not a bot").
Clipora doesn't try to get around that. Instead, YouTube streams are recorded from a normal home internet
connection: an always-on PC runs a GitHub "self-hosted runner", and GitHub hands it the YouTube jobs.
Twitch streams keep running on GitHub's own servers.

## Windows PC setup (about 10 minutes, once)

1. Install **Docker Desktop** (docker.com) and start it. Turn on "Start Docker Desktop when you sign in".
2. On GitHub open **blackbargoat/stream-clipper → Settings → Actions → Runners → New self-hosted runner**,
   choose **Linux**, and copy the token from the `./config.sh ... --token XXXX` line (it lasts one hour).
3. In PowerShell run (paste your token where it says TOKEN):

       docker run -d --name clipora-runner --restart always `
         -e REPO_URL=https://github.com/blackbargoat/stream-clipper `
         -e RUNNER_TOKEN=TOKEN -e RUNNER_NAME=home-pc -e LABELS=home `
         -e RUNNER_SCOPE=repo -e DISABLE_AUTO_UPDATE=1 `
         myoung34/github-runner:latest

4. Back on the Runners page, **home-pc** should show as Idle (green).
5. Tell Claude it's up. It sets the repo variable `HOME_RUNNER=1`, and from then on YouTube streams and the
   watcher run at home automatically. Keep the PC on and set it to never sleep.
