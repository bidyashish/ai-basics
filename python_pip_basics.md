# Python & pip Basics — A Beginner's Survival Guide (macOS)

> A practical reference for new Python developers on macOS, covering pip, paths, virtual environments, and how to debug common issues.

---

## 1. What is pip?

**pip** = **P**ackage **I**nstaller for **P**ython

It's the tool that lets you install third-party libraries (like `numpy`, `requests`, `flask`, etc.) that don't come built into Python.

```bash
# Basic syntax
pip install <package-name>       # Install a package
pip uninstall <package-name>     # Remove a package
pip list                         # See all installed packages
pip show <package-name>          # Details about a specific package
pip freeze                       # List packages in requirements.txt format
pip install -r requirements.txt  # Install all packages from a file
```

### Useful pip Commands

| Command | What it does |
|---|---|
| `pip install numpy` | Install latest numpy |
| `pip install numpy==2.0.0` | Install a specific version |
| `pip install numpy>=2.0` | Install version 2.0 or higher |
| `pip install --upgrade numpy` | Upgrade to latest version |
| `pip install --upgrade pip` | Upgrade pip itself |
| `pip cache purge` | Clear pip's download cache |

---

## 2. Understanding Python & pip Paths on macOS

### Where is Python?

macOS with Homebrew typically has **multiple Pythons**. This is the #1 source of confusion.

```bash
# Find which python you're using
which python3
# Example output: /opt/homebrew/bin/python3  (Homebrew system Python)

# Find ALL pythons on your system
where python3
# or
ls -la /opt/homebrew/bin/python3*
```

### Where is pip?

pip lives **alongside** the Python it belongs to:

```bash
which pip3
# Example output: /opt/homebrew/bin/pip3  (system pip — DON'T use this for packages)

# Always safer to use:
python3 -m pip install <package>
# This guarantees you're using the pip that matches your python3
```

### The Path Hierarchy

```
/opt/homebrew/bin/python3          ← Homebrew's system Python (PROTECTED)
/opt/homebrew/bin/pip3             ← Homebrew's system pip (PROTECTED)

/Users/ashish/Developer/code-run/venv/bin/python   ← Your venv Python ✅
/Users/ashish/Developer/code-run/venv/bin/pip       ← Your venv pip ✅
```

> [!IMPORTANT]
> On modern macOS (with Homebrew Python 3.12+), the system Python is **externally managed** — pip refuses to install packages into it. This is the `externally-managed-environment` error you saw. **Always use a virtual environment.**

---

## 3. Virtual Environments (venv) — The Golden Rule

### Why?

- Keeps each project's dependencies **isolated**
- Avoids breaking your system Python
- Avoids version conflicts between projects

### How to Create & Use

```bash
# 1. Navigate to your project
cd /Users/ashish/Developer/code-run

# 2. Create a virtual environment (one-time)
python3 -m venv venv

# 3. Activate it
# ┌──────────────────────────────────────────┐
# │  Shell        │  Activation Command      │
# ├──────────────────────────────────────────┤
# │  bash/zsh     │  source venv/bin/activate │
# │  fish         │  source venv/bin/activate.fish │
# │  Windows CMD  │  venv\Scripts\activate   │
# └──────────────────────────────────────────┘

source venv/bin/activate.fish   # ← You use fish shell

# 4. Now pip installs go into the venv (safe!)
pip install numpy

# 5. Run your script
python jo.py

# 6. Deactivate when done
deactivate
```

### How to Tell if venv is Active

When activated, your prompt shows the venv name:

```
(venv) ashish@ashishs-MacBook-Pro code-run>
```

You can also verify:

```bash
which python
# Should show: /Users/ashish/Developer/code-run/venv/bin/python  ✅
# NOT: /opt/homebrew/bin/python3  ❌
```

> [!TIP]
> **Quick sanity check:** If `which python` or `which pip` points to `/opt/homebrew/...`, your venv is NOT active. Activate it first!

---

## 4. Common Errors & How to Debug

### Error 1: `externally-managed-environment`

```
error: externally-managed-environment
× This environment is externally managed
```

**Cause:** You're trying to install into Homebrew's system Python.

**Fix:** Use a virtual environment (see Section 3 above).

```bash
python3 -m venv venv
source venv/bin/activate.fish
pip install <package>
```

---

### Error 2: `ModuleNotFoundError: No module named 'xyz'`

```
ModuleNotFoundError: No module named 'numpy'
```

**Cause:** The package isn't installed in the Python you're running.

**Debug steps:**

```bash
# Step 1: Check which python is running
which python

# Step 2: Check if the package is installed in THAT python
pip list | grep numpy

# Step 3: If not installed, install it
pip install numpy

# Step 4: Make sure you run with the SAME python
python your_script.py    # ← Use 'python', not 'python3' or '/usr/bin/python3'
```

> [!WARNING]
> The most common cause: you installed numpy with one Python but ran the script with a **different** Python. Always verify with `which python`.

---

### Error 3: `command not found: pip`

```bash
# Fix: Use python3 -m pip instead
python3 -m pip install numpy
```

---

### Error 4: Wrong Python Version

```bash
# Check your version
python --version
# Python 3.14.0

# If you need a specific version, install with pyenv:
brew install pyenv
pyenv install 3.12.0
pyenv local 3.12.0
```

---

## 5. Debugging Workflow Cheat Sheet

When something doesn't work, follow this checklist **in order**:

```
┌─────────────────────────────────────────────────┐
│          Python Debugging Flowchart             │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Is your venv activated?                     │
│     → which python                              │
│     → Should show .../venv/bin/python            │
│                                                 │
│  2. Is the package installed?                   │
│     → pip list | grep <package>                 │
│                                                 │
│  3. Are you running the right python?           │
│     → python script.py  (not python3 or full    │
│       path to system python)                    │
│                                                 │
│  4. Read the error message carefully            │
│     → Python errors read BOTTOM to TOP          │
│     → The last line = the actual error          │
│     → Lines above = the call stack (where it    │
│       happened)                                 │
│                                                 │
│  5. Google the exact error message              │
│     → Copy the last line of the traceback       │
│     → Paste into Google/StackOverflow           │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 6. Project Setup Template

For every new Python project, follow this pattern:

```bash
# Create project directory
mkdir ~/Developer/my-new-project
cd ~/Developer/my-new-project

# Create venv
python3 -m venv venv

# Activate (fish shell)
source venv/bin/activate.fish

# Install packages
pip install numpy pandas matplotlib

# Save dependencies
pip freeze > requirements.txt

# Write your code
touch main.py

# Run
python main.py
```

### Restoring a Project Later

```bash
cd ~/Developer/my-new-project
source venv/bin/activate.fish
pip install -r requirements.txt   # Reinstall all saved dependencies
python main.py
```

---

## 7. Quick Reference Card

| Task | Command |
|---|---|
| Create venv | `python3 -m venv venv` |
| Activate (fish) | `source venv/bin/activate.fish` |
| Activate (bash/zsh) | `source venv/bin/activate` |
| Deactivate | `deactivate` |
| Install package | `pip install <pkg>` |
| Check which python | `which python` |
| Check installed packages | `pip list` |
| Save dependencies | `pip freeze > requirements.txt` |
| Install from file | `pip install -r requirements.txt` |
| Check Python version | `python --version` |
| Run a script | `python script.py` |

> [!NOTE]
> **Your current setup:**
> - Python: `3.14` (Homebrew)
> - Shell: `fish`
> - Venv location: `/Users/ashish/Developer/code-run/venv`
> - Activation: `source venv/bin/activate.fish`
