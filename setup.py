#!/usr/bin/env python3
import os
import subprocess
import venv
from pathlib import Path
from security import safe_command

def main():
    # Create data directory structure
    data_dir = Path("data")
    data_subdirs = ["raw", "processed"]
    
    for subdir in data_subdirs:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("Created data directory structure")

    # Create virtual environment
    venv_dir = Path(".venv")
    venv.create(venv_dir, with_pip=True)
    print("Created virtual environment")

    # Determine activation script based on OS
    if os.name == "nt":  # Windows
        activate_script = venv_dir / "Scripts" / "activate.bat"
        activate_cmd = str(activate_script)
    else:  # Unix-like
        activate_script = venv_dir / "bin" / "activate"
        activate_cmd = f"source {activate_script}"

    # Activate venv and install requirements
    install_cmd = f"{activate_cmd} && pip install -r requirements.txt"
    
    try:
        safe_command.run(subprocess.run, install_cmd, shell=False, check=True)
        print("Installed requirements successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return 1

    print("Setup completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
