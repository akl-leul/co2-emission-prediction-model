# CO2 Charcoal Analyzer - Desktop Version

This is the desktop version of the CO2 Charcoal Emission Analyzer application.

## Prerequisites

1. Python 3.7 or higher
2. pip (Python package manager)
3. NSIS (Nullsoft Scriptable Install System) - Required only for creating the installer

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements-desktop.txt
   ```

2. **Build the Executable**:
   Run the build script:
   ```bash
   python build.py
   ```
   This will:
   - Install required dependencies
   - Create a standalone executable in the `dist` folder
   - Optionally create an installer (requires NSIS)

## Creating the Installer

To create an installer:

1. Install NSIS from: https://nsis.sourceforge.io/Download
2. Add NSIS to your system PATH
3. Run the build script and choose 'y' when prompted to create an installer
4. The installer will be created at `dist/CO2CharcoalAnalyzer_Setup.exe`

## Running the Application

### From Source
```bash
python desktop_app.py
```

### From Executable
Run `dist/CO2CharcoalAnalyzer/CO2CharcoalAnalyzer.exe`

### From Installer
Run the installer and follow the on-screen instructions.

## Files

- `desktop_app.py` - Main application entry point
- `app.py` - Original Streamlit application
- `build.py` - Build script for creating executables and installers
- `requirements-desktop.txt` - Python dependencies
- `ai_learning_history.json` - AI learning data (will be created if not exists)
- `ai_full_data_history.json` - Full data history (will be created if not exists)

## Distribution

To distribute your application:

1. For simple distribution, share the `CO2CharcoalAnalyzer.exe` file from the `dist` folder
2. For a more professional installation, distribute the `CO2CharcoalAnalyzer_Setup.exe` installer

## Notes

- The application requires an internet connection for the first run to download any additional dependencies
- All user data is stored in the application's directory
- Make sure to include all data files (`*.json`) when distributing the application
