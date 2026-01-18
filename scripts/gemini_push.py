import argparse
import subprocess
import sys
import os
import tempfile

def run_command(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automated Gemini Commit & Push")
    parser.add_argument("--request", required=True, help="The user's original request (or rephrased).")
    parser.add_argument("--summary", required=True, help="Summary of changes made.")
    
    args = parser.parse_args()

    # 1. Add all changes
    print("Staging all changes...")
    run_command("git add .")

    # 2. Construct Commit Message
    commit_msg = f"""{args.request}

Summary:
{args.summary}

Automated commit by Gemini CLI.
"""
    
    # 3. Commit (Initial)
    print("Committing (Initial)...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tf:
        tf.write(commit_msg)
        temp_file_path = tf.name

    try:
        run_command(f'git commit -F "{temp_file_path}"')
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # 4. Generate/Update Log
    print("Updating Gemini_command.md...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_script = os.path.join(script_dir, "generate_log.py")
    
    if os.path.exists(log_script):
        subprocess.run([sys.executable, log_script], check=True)
    else:
        print(f"Warning: {log_script} not found. Skipping log update.")

    # 5. Amend Commit with Log Update
    print("Amending commit with log update...")
    run_command("git add Gemini_command.md")
    run_command("git commit --amend --no-edit")

    # 6. Push
    print("Pushing to upstream...")
    run_command("git push")

    print("Success! Changes committed and pushed.")

if __name__ == "__main__":
    main()