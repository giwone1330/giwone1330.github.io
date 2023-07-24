---
title: Setting up Obsidian-git environment for cross-platform notetaking
date: 2023-07-24 00:00:00 +0900
category: [miscellaneous, utility]
tags: [] # TAG names shouldalways be lowercase
---

## Obsidian?

Obsidian is a powerful notetaking software that supports most of the markdown syntax with it's unique graph view.
Personally I have been using Notion for the last few months, but the graph view feature of Obsidian lead me to try it.
However, unlike Notion Obsidian doesn't provide cloud-sync from the go, unless you pay a certain subscription fee.
But by using the powerful plugins provieded by the community, I was able to make obsidian work across platforms.

## Ubuntu
### Installation

Always check for the latest releases from homepage.

[Download Obsidian](https://obsidian.md/download)

And for Obsidian-git to work, Obsidian should be installed with AppImage!

```bash
# Downlaod
wget https://github.com/obsidianmd/obsidian-releases/releases/download/v1.3.5/Obsidian-1.3.5.AppImage
chmod u+x Obsidian-1.3.5.AppImage

# Run
./Obsidian-1.3.5.AppImage
```
{: .terminal}

### Authentication

```bash
git config --global credential.helper libsecret
sudo apt install libsecret-1-0 libsecret-1-dev make gcc

# NOTE: This changes your global config, in case you don't want that you can omit the `--global` and execute it in your existing git repository.
git config --global credential.helper \
   /usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret
```
{: .terminal}

### Make Github Repo

Just make a private github repo that you will use to archive obsidian vault.

You might also need github authentification token with basic permissions such as "Repo"...etc. Keep then well.

You can use your github id and this token to authenticate if the terminal asks you to do so.


### Obsidian Git

From `Obsidian > Settings > Options > Community plugins` browse and download Obsidian Git, enable them, and go to `Obsidian > Settings > Community plugings > Obsidian Git`.

Here modify as you prefer, and set the `Vault backup interval (minutes)` to anything other than 0. This will automate github syncing.

Now from the main window of Obsidian, `Ctrl + p` or `Cmd + p` to bring up command pallet. Here search and run `Obsidina Git: Clone an existing remote repo`, give your repo link such as`https://github.com/<username>/<reponame>.git` and proceed.

Here check the terminal if there's no response. It might ask you your github autorization.

The ubuntu part is almost done! You can monitor the git manually by using the command `Obsidian Git: Open source control view` Check for the Obsidian Git pluging settings for any problems.

## Android
### Installation and Setup

Install Obsidian from google play store or any app stores you use.
**Then Open a vault from the parent folder of your desired actual vault.**

Browse and downlaod Obsidian Git, and enable it. This is same as ubuntu.

Fill in git server and github access token.
And don't touch "Advanced".

From the main obsidian interface, use command pallet to run `Obsidina Git: Clone an existing remote repo`

Now specify the gitub repo base path by typing in your github repo name eg.`my_obsidian_repo`.

Follow the process, and it will clone your repo. When done, open the vault you downloaded.

The manual control of the git sources are done in the same way as ubuntu.

If there's any problem, check plugin settings and restart the app or the device and try again.

Also try to keep only one Obsidian at a time. The periodic sync make it kind of buggy.

## Other plugin suggestions
- Advanced Slides
- Day Planner
- Dice Roller
- Excalidraw
- Kanban
- Linter
- TagFolder
- Version History Diff


