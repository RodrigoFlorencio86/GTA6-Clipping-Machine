# Acceptable use

GTA6 Clipping Machine is a tool for creators who want to find good moments in **their own**
gameplay recordings. These rules are part of the project. Some are enforced in code, and the
test suite checks that they stay in place.

## You may

- Import gameplay that **you recorded yourself** from a legitimate copy of the game.
- Import footage that someone else recorded **if they have authorized you** to use it (for
  example, a teammate's capture or a client you edit for).
- Export clips and post them yourself, on the platforms you choose, after you have reviewed them.

## You may not

- Import pirated, leaked or otherwise unauthorized footage, including pre-release material.
- Import other people's videos (streams, YouTube uploads, TikToks) without their permission.
- Use the project to download content from any platform. The app accepts **local files only**
  and rejects links on purpose. Pull requests that add URL downloaders will not be accepted.
- Use it to promote cheats, exploits, mods that break the game's terms, or any illegal content.
- Present clips as official material from the game's publisher.

## What the app does and does not do

| Rule | How it is enforced |
|---|---|
| Local files only | `video_validator.looks_like_url` rejects links; there is no download code; `tests/test_guardrails.py` fails if networking or downloader libraries are imported or declared as dependencies. |
| Ownership confirmation | Import is refused until you tick the confirmation box (UI) or pass `--i-own-this-footage` (CLI). The confirmation time is stored with the video. |
| Human approval | A clip starts as `candidate`. The final MP4 export is refused unless a person has moved it to `approved`. |
| No automatic publishing | There is no publisher or uploader module. `published` is only a note you add after posting the file yourself. |
| Local processing | Video, audio, frames and embeddings never leave your computer. The only network access is the one-time model download, which `GCM_OFFLINE=1` blocks afterwards. |
| No telemetry | Hugging Face and Streamlit usage reporting are switched off. The metrics on the Status page are stored in a local SQLite file and never transmitted. |

## Rights and platform rules are your responsibility

You are responsible for the rights to everything you import and publish. Game publishers have
their own policies on fan content and monetization, and they may change them. Read the current
policy of the game's publisher (for GTA, the
[Rockstar Games policy on posting copyrighted material](https://support.rockstargames.com/articles/7bNaeoMFTV0iUDGhStTXvz/policy-on-posting-copyrighted-Rockstar-Games-material))
and the terms of each platform you post to before you publish or monetize clips.

## No promises about reach

The highlight score ranks candidates so you can review the most promising ones first. It
does **not** predict views, engagement, monetization or virality, and nothing in this project
should be read as such a promise.

## Trademarks

This is an independent open-source project. It is not affiliated with, endorsed by or
sponsored by Rockstar Games, Take-Two Interactive, Google, TikTok, Instagram or YouTube.
"Grand Theft Auto" and "GTA" are trademarks of their respective owners and are used here only
to describe the kind of gameplay the tool is designed for. The tool works with footage from
any game.
