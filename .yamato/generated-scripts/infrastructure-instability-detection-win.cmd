@echo on
rem This is an auto-generated script. Do not edit manually!

if exist "%TEMP%\BugReporterCrashReportJson"  for /f "delims=" %%i in ('dir /b /a-d "%TEMP%\BugReporterCrashReportJson\CrashReport_*.json"') do curl -X POST --connect-timeout 5 --max-time 10 -H "Content-Type: application/json" -T "%TEMP%\BugReporterCrashReportJson\%%i" "https://internal-crash-collector.prd.cds.internal.unity3d.com/api/crash" || echo Failed to upload %%i. Ignoring...
IF "%YAMATO_COMMAND_BLOCK_SUCCESS%"=="1" exit /b 0
curl -fs "https://artifactory-slo.bf.unity3d.com/artifactory/automation-and-tooling/infrastructure-instability-detection/standalone/1.2.2/windows.zip" --output "infrastructure_instability_detection_standalone.zip" --retry 5
IF EXIST "infrastructure_instability_detection" rmdir /s /q infrastructure_instability_detection
powershell.exe -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('infrastructure_instability_detection_standalone.zip', '.'); }" && DEL "infrastructure_instability_detection_standalone.zip"
curl -fs "https://artifactory-slo.bf.unity3d.com/artifactory/automation-and-tooling/infrastructure-instability-detection/patterns.zip" --output patterns.zip --retry 5
IF EXIST "patterns" rmdir /s /q patterns
powershell.exe -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('patterns.zip', '.'); }" && DEL "patterns.zip"
infrastructure_instability_detection
exit /b 0
