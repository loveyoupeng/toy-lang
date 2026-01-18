import subprocess
import sys
import os
from datetime import datetime

LOG_FILE = "Gemini_command.md"

def get_latest_commit_info():
    try:
        # Get hash, date, author, and full body
        # %H: commit hash
        # %cd: commit date
        # %an: author name
        # %B: raw body (subject and body)
        cmd = ["git", "log", "-1", "--pretty=format:%H%n%cd%n%an%n%B"]
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        lines = result.stdout.strip().split('\n')
        if not lines or len(lines) < 4:
            return None
        commit_hash = lines[0]
        date = lines[1]
        author = lines[2]
        message = "\n".join(lines[3:])
        return commit_hash, date, author, message
    except subprocess.CalledProcessError:
        return None

def update_log(commit_hash, date, author, message):
    entry = f"""
## Commit: {commit_hash}
**Date:** {date}
**Author:** {author}

**Message:**
{message}

---
"""
    # Append to log file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"Updated {LOG_FILE} with commit {commit_hash}")

def main():
    info = get_latest_commit_info()
    if info:
        update_log(*info)
    else:
        print("No commits found or error reading git log.")

if __name__ == "__main__":
    main()