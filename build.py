import os
import shutil
import subprocess
import sys
from pathlib import Path

def build_executable():
    print("Building executable...")
    
    # Create build directory
    build_dir = Path("build")
    dist_dir = Path("dist")
    
    # Clean previous builds
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # PyInstaller command
    pyinstaller_cmd = [
        'pyinstaller',
        '--name=CO2CharcoalAnalyzer',
        '--onefile',
        '--windowed',
        '--add-data=ai_learning_history.json;.',
        '--add-data=ai_full_data_history.json;.',
        'desktop_app.py'
    ]
    
    # Run PyInstaller
    subprocess.check_call(pyinstaller_cmd)
    
    print("\nBuild completed successfully!")
    print(f"Executable created at: {dist_dir / 'CO2CharcoalAnalyzer.exe'}")

def create_installer():
    print("\nCreating installer...")
    
    # Create NSIS script for the installer
    nsis_script = """
    ; NSIS Installer Script
    !include "MUI2.nsh"
    
    ; General settings
    Name "CO2 Charcoal Analyzer"
    OutFile "dist/CO2CharcoalAnalyzer_Setup.exe"
    InstallDir "$PROGRAMFILES\\CO2CharcoalAnalyzer"
    
    ; Request admin privileges
    RequestExecutionLevel admin
    
    ; Interface settings
    !define MUI_ABORTWARNING
    !define MUI_ICON "app.ico"
    
    ; Pages
    !insertmacro MUI_PAGE_DIRECTORY
    !insertmacro MUI_PAGE_INSTFILES
    !insertmacro MUI_PAGE_FINISH
    
    ; Languages
    !insertmacro MUI_LANGUAGE "English"
    
    ; Installer sections
    Section "MainSection" SEC01
        SetOutPath "$INSTDIR"
        File /r "dist\\CO2CharcoalAnalyzer\\*.*"
        
        ; Create start menu shortcuts
        CreateDirectory "$SMPROGRAMS\\CO2CharcoalAnalyzer"
        CreateShortCut "$SMPROGRAMS\\CO2CharcoalAnalyzer\\CO2CharcoalAnalyzer.lnk" "$INSTDIR\\CO2CharcoalAnalyzer.exe"
        CreateShortCut "$DESKTOP\\CO2CharcoalAnalyzer.lnk" "$INSTDIR\\CO2CharcoalAnalyzer.exe"
        
        ; Add uninstaller
        WriteUninstaller "$INSTDIR\\uninstall.exe"
        WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CO2CharcoalAnalyzer" \
                        "DisplayName" "CO2 Charcoal Analyzer"
        WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CO2CharcoalAnalyzer" \
                        "UninstallString" "$\"$INSTDIR\\uninstall.exe$\""
    SectionEnd
    
    Section "Uninstall"
        ; Remove files and directories
        RMDir /r "$INSTDIR"
        RMDir /r "$SMPROGRAMS\\CO2CharcoalAnalyzer"
        Delete "$DESKTOP\\CO2CharcoalAnalyzer.lnk"
        
        ; Remove uninstaller registry keys
        DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CO2CharcoalAnalyzer"
    SectionEnd
    """
    
    # Write NSIS script to file
    with open("installer.nsi", "w") as f:
        f.write(nsis_script)
    
    # Create the installer using NSIS (must be installed on the system)
    try:
        subprocess.check_call(['makensis', 'installer.nsi'])
        print("\nInstaller created successfully!")
        print(f"Installer path: {Path('dist') / 'CO2CharcoalAnalyzer_Setup.exe'}")
    except FileNotFoundError:
        print("\nNSIS (makensis) not found. Please install NSIS and add it to your PATH.")
        print("You can download it from: https://nsis.sourceforge.io/Download")

if __name__ == "__main__":
    # Install required packages
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-desktop.txt'])
    
    # Build the executable
    build_executable()
    
    # Ask if user wants to create an installer
    if input("\nDo you want to create an installer? (y/n): ").lower() == 'y':
        create_installer()
    
    print("\nBuild process completed!")
