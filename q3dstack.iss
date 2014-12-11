; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!
; 
; Q3DStack is distributed under the BSD License.
; See the accompanying file LICENSE for details.
; 
; Copyright (c) 2009-2014 Broad Institute
; All rights reserved.
; 
; Please see the AUTHORS file for credits.
; 
; Website: http://www.cellprofiler.org

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{568199af-3e01-4cd0-bfa4-0b071a1dd889}
AppName=Q3DStack
AppVerName=Q3DStack 1.0
OutputBaseFilename=Q3DStack_1.0
AppPublisher=Broad Institute
AppPublisherURL=http://www.cellprofiler.org
AppSupportURL=http://github.com/LeeKamentsky/q3dstack
AppUpdatesURL=http://www.cellprofiler.org
DefaultDirName={pf64}\Q3DStack
DefaultGroupName=Q3DStack
OutputDir=.\output
SetupIconFile=.\q3dstack.ico
Compression=lzma
SolidCompression=yes
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: ".\dist\q3dstack.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: ".\dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[InstallDelete]
Type: files; Name: {app}\jars\*.jar

[Icons]
Name: "{group}\Q3DStack"; Filename: "{app}\q3dstack.exe"; WorkingDir: "{app}"
Name: "{group}\{cm:ProgramOnTheWeb,Q3DStack}"; Filename: "http://github.com/LeeKamentsky/q3dstack"
Name: "{group}\{cm:UninstallProgram,Q3DStack}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\Q3DStack"; Filename: "{app}\q3dstack.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\q3dstack.exe"; Description: "{cm:LaunchProgram,q3dstack}"; Flags: nowait postinstall skipifsilent

