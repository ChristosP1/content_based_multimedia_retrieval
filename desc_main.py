import subprocess
import sys
import time

if __name__ == "__main__":
    # List of scripts to run
    scripts = ['desc_global.py', 'desc_local.py']

    # Use the same Python interpreter as the current environment
    python_exec = sys.executable

    # Run each script
    for script in scripts:
        subprocess.run([python_exec, script])
        time.sleep(1)
