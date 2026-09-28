param (
    [string]$InputPath = "C:\Users\victo\.gemini\antigravity\scratch\heart-disease-prediction\241VMTR02058_Benedict_Baah_Defense_Presentation.pptx",
    [string]$OutputPath = "C:\Users\victo\.gemini\antigravity\scratch\heart-disease-prediction\241VMTR02058_Benedict_Baah_Defense_Presentation.pdf"
)

try {
    Write-Host "Starting PowerPoint COM automation..."
    $ppt = New-Object -ComObject PowerPoint.Application
    # Open presentation (ReadOnly, Untitled, WithWindow)
    $pres = $ppt.Presentations.Open($InputPath, 0, 0, 0)
    # 32 = ppSaveAsPDF
    $pres.SaveAs($OutputPath, 32)
    $pres.Close()
    $ppt.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($pres) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
    Write-Host "Successfully converted presentation to PDF: $OutputPath"
} catch {
    Write-Error "Failed to convert PPTX to PDF: $_"
    exit 1
}
