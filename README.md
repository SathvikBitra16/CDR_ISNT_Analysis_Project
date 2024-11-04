# CDR_ISNT Analysis Project

## Overview

This project is designed to analyze Call Detail Records (CDRs) using Python and Flask. It provides a web interface for interacting with the analysis tools.

## Prerequisites

- Python 3.x installed on your system
- Visual Studio Code (VSCode) installed
- Git (optional, for version control)

## Setting Up the Environment

Follow these steps to set up the Python environment using the existing `.venv` directory in the project.

### Step 1: Open the Project in VSCode

1. Launch Visual Studio Code.
2. Open the project directory by navigating to `File` > `Open Folder...` and selecting the folder containing your project.

### Step 2: Configure VSCode for Python

1. **Install the Python extension for VSCode** if you haven't already:
   - Open the Extensions view by pressing `Ctrl + Shift + X`.
   - Search for "Python" and install the extension published by Microsoft.

2. **Set the Python interpreter to your virtual environment:**
   - Press `Ctrl + Shift + P` to open the Command Palette.
   - Type and select **Python: Select Interpreter**.
   - Choose the interpreter from your virtual environment (it should look like `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux).

### Step 3: Install necessary Python libraries

1. **Install your libraries:**
   
   ```bash
   pip install flask flask-cors

### Step 4: Run Your Flask Application

1. **Open the terminal in VSCode:**
   - Navigate to **Terminal** > **New Terminal** or use the shortcut `` Ctrl + ` ``.

2. **Ensure your virtual environment is activated in the terminal.**
   - If it isn’t activated automatically, you can activate it again using the command from Step 2 of the setup instructions.

3. **Run your Flask application:**
   ```bash
   python app.py

