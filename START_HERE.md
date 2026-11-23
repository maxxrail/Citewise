# Start here (for Maxx)

## One-time setup (Windows)
1. Install: **Docker Desktop**, **uv** (`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`), **Node.js 20+** (for Module 8), Git.
2. In GitHub Desktop: *File → New repository* → name `citewise` → pick a local path. Copy every file from this kit into that folder (including the hidden `.claude` folder) → commit "project kit" → Publish (public repo).
3. Open the folder in VS Code, open a terminal there and run `claude`.

> **Can't see the `.claude` folder?** Windows hides folders that start with a dot. In File Explorer: *View → Show → Hidden items*. Make sure it gets copied into the repo; it holds the commands, subagents, permissions and hooks.

## What's in `.claude/`
| File | What it does |
|---|---|
| `settings.json` | Pre-approves safe commands (uv, pytest, ruff, git commit), asks before `git push` or real paid scans, blocks reading `.env`, defaults to mock mode, auto-formats code after edits |
| `commands/next.md` | `/next`: builds the next unfinished module (easiest way to go) |
| `commands/build-module.md` | `/build-module 03`: builds a specific module |
| `commands/review-module.md` | `/review-module 03`: interviewer-style review + fixes |
| `commands/status.md` | `/status`: where things stand, what to type next |
| `commands/fix-tests.md` | `/fix-tests`: gets tests green without cheating |
| `commands/interview-prep.md` | `/interview-prep`: quizzes you on your own code |
| `commands/ship-check.md` | `/ship-check`: final checklist before you apply |
| `agents/*.md` | Subagents Claude Code calls itself: docs checker, test writer, eval designer, code reviewer |

Personal overrides go in `.claude/settings.local.json` (already git-ignored).

## Build a module
In Claude Code, type:

```
/build-module 00
```

When it finishes and the checklist is all ✅:

```
/review-module 00
```

Or just type `/next` every time and it figures out which module is up.

Then check the work yourself (it gives you the commands), commit/push in GitHub Desktop, and move on to `01`, `02`… through `08`. Start a **fresh Claude Code session** (`/clear`) for each module; the specs and PROGRESS.md carry the context.

## What's in `.claude/` (hidden folder: turn on *View → Show → Hidden items* in File Explorer)
| File | What it does |
|---|---|
| `settings.json` | Permissions: Claude Code can run uv/pytest/ruff/docker/git commit without asking you every time; it **asks** before `git push` or a real (paid) scan; it can't read your `.env` |
| `commands/build-module.md` | `/build-module 03` builds a module from its spec |
| `commands/review-module.md` | `/review-module 03` runs an interviewer-style review, fixes issues, writes interview prep |
| `commands/status.md` | `/status` shows where you are and the next command to type |
| `commands/fix-tests.md` | `/fix-tests` gets tests green without cheating |
| `commands/interview-prep.md` | `/interview-prep` mock interview on your own code |
| `agents/code-reviewer.md` | Subagent that reviews code with a fresh eye |
| `agents/eval-designer.md` | Subagent that drafts golden datasets, rubrics and red-team cases |

Your own personal overrides go in `.claude/settings.local.json` (git-ignored).

## Things only you can do
- **API keys** (from Module 2): put them in `.env`. Budget ~$10–20 total for real scans across the whole project. Set spending limits on each provider dashboard.
- **Label data** (Modules 3 and 6): review ~20–60 golden examples. This is what makes the eval numbers believable in an interview. Don't skip it.
- **One real scan** (Module 8): run a live scan on a real brand so the write-up has real numbers.
- **Deploy** (Module 8): create Railway/Fly + Vercel accounts and follow `docs/DEPLOY.md`.

## Suggested pace
| Week | Modules |
|---|---|
| 1 | 0, 1, 2 |
| 2 | 3, 4 |
| 3 | 5, 6 |
| 4 | 7, 8 + apply |

You can apply after Module 3: "core loop works, here's the repo, more shipping weekly" is a good signal for a startup too.
