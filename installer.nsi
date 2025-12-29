
    ; NSIS Installer Script
    !include "MUI2.nsh"
    
    ; General settings
    Name "CO2 Charcoal Analyzer"
    OutFile "dist/CO2CharcoalAnalyzer_Setup.exe"
    InstallDir "$PROGRAMFILES\CO2CharcoalAnalyzer"
    
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
        File /r "dist\CO2CharcoalAnalyzer\*.*"
        
        ; Create start menu shortcuts
        CreateDirectory "$SMPROGRAMS\CO2CharcoalAnalyzer"
        CreateShortCut "$SMPROGRAMS\CO2CharcoalAnalyzer\CO2CharcoalAnalyzer.lnk" "$INSTDIR\CO2CharcoalAnalyzer.exe"
        CreateShortCut "$DESKTOP\CO2CharcoalAnalyzer.lnk" "$INSTDIR\CO2CharcoalAnalyzer.exe"
        
        ; Add uninstaller
        WriteUninstaller "$INSTDIR\uninstall.exe"
        WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\CO2CharcoalAnalyzer"                         "DisplayName" "CO2 Charcoal Analyzer"
        WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\CO2CharcoalAnalyzer"                         "UninstallString" "$"$INSTDIR\uninstall.exe$""
    SectionEnd
    
    Section "Uninstall"
        ; Remove files and directories
        RMDir /r "$INSTDIR"
        RMDir /r "$SMPROGRAMS\CO2CharcoalAnalyzer"
        Delete "$DESKTOP\CO2CharcoalAnalyzer.lnk"
        
        ; Remove uninstaller registry keys
        DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\CO2CharcoalAnalyzer"
    SectionEnd
    