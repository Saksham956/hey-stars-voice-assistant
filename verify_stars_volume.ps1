param([switch]$TestActions)
# M02-R3 diagnostic only. Does not run STARS, listen, synthesize or call Gemini.
# Uses Windows endpoint readback, independent of Vosk and intent matching.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not ('StarsVolumeProbe' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
[ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IStarsEnumerator {
    [PreserveSig] int EnumAudioEndpoints(int flow, uint mask, out IntPtr devices);
    [PreserveSig] int GetDefaultAudioEndpoint(int flow, int role, out IStarsDevice device);
}
[ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IStarsDevice {
    [PreserveSig] int Activate(ref Guid iid, uint context, IntPtr parameters, out IStarsEndpoint endpoint);
}
[ComImport, Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IStarsEndpoint {
    [PreserveSig] int RegisterControlChangeNotify(IntPtr callback);
    [PreserveSig] int UnregisterControlChangeNotify(IntPtr callback);
    [PreserveSig] int GetChannelCount(out uint channels);
    [PreserveSig] int SetMasterVolumeLevel(float level, IntPtr context);
    [PreserveSig] int SetMasterVolumeLevelScalar(float level, IntPtr context);
    [PreserveSig] int GetMasterVolumeLevel(out float level);
    [PreserveSig] int GetMasterVolumeLevelScalar(out float level);
    [PreserveSig] int SetChannelVolumeLevel(uint channel, float level, IntPtr context);
    [PreserveSig] int SetChannelVolumeLevelScalar(uint channel, float level, IntPtr context);
    [PreserveSig] int GetChannelVolumeLevel(uint channel, out float level);
    [PreserveSig] int GetChannelVolumeLevelScalar(uint channel, out float level);
    [PreserveSig] int SetMute([MarshalAs(UnmanagedType.Bool)] bool mute, IntPtr context);
    [PreserveSig] int GetMute([MarshalAs(UnmanagedType.Bool)] out bool mute);
}
public class StarsVolumeState { public float Level; public bool Muted; }
public static class StarsVolumeProbe {
    static IStarsEndpoint Open() {
        IStarsEnumerator enumerator = null;
        IStarsDevice device = null;
        try {
            enumerator = (IStarsEnumerator)Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")));
            Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(0, 0, out device));
            Guid iid = new Guid("5CDF2C82-841E-4546-9722-0CF74078229A");
            IStarsEndpoint endpoint;
            Marshal.ThrowExceptionForHR(device.Activate(ref iid, 23, IntPtr.Zero, out endpoint));
            return endpoint;
        } finally {
            if (device != null) Marshal.ReleaseComObject(device);
            if (enumerator != null) Marshal.ReleaseComObject(enumerator);
        }
    }
    public static StarsVolumeState Read() {
        IStarsEndpoint endpoint = Open();
        try {
            StarsVolumeState state = new StarsVolumeState();
            Marshal.ThrowExceptionForHR(endpoint.GetMasterVolumeLevelScalar(out state.Level));
            Marshal.ThrowExceptionForHR(endpoint.GetMute(out state.Muted));
            return state;
        } finally { Marshal.ReleaseComObject(endpoint); }
    }
    public static void Restore(float level, bool mute) {
        IStarsEndpoint endpoint = Open();
        try {
            Marshal.ThrowExceptionForHR(endpoint.SetMasterVolumeLevelScalar(level, IntPtr.Zero));
            Marshal.ThrowExceptionForHR(endpoint.SetMute(mute, IntPtr.Zero));
        } finally { Marshal.ReleaseComObject(endpoint); }
    }
}
'@
}
$starsInitialState = [StarsVolumeProbe]::Read()
$starsResults = [ordered]@{ scope='Default Windows render endpoint, eConsole; no speech recognition'; initial_level=$starsInitialState.Level; initial_muted=$starsInitialState.Muted; actions_tested=[bool]$TestActions; tests=@() }
if ($TestActions) {
    try {
        foreach ($starsDirection in @('volume_up','volume_down')) {
            [StarsVolumeProbe]::Restore($starsInitialState.Level, $starsInitialState.Muted)
            $starsBefore = [StarsVolumeProbe]::Read()
            $starsAtLimit = ($starsDirection -eq 'volume_up' -and $starsBefore.Level -ge 0.9999) -or ($starsDirection -eq 'volume_down' -and $starsBefore.Level -le 0.0001)
            if ($starsAtLimit) {
                $starsResults.tests += [ordered]@{intent=$starsDirection; status='not_tested_at_volume_limit'; before=$starsBefore.Level}
                continue
            }
            $starsReply = & '.\.venv\Scripts\python.exe' -c 'import sys; from local_commands import LocalRouter,Intent; print(LocalRouter().execute(Intent(sys.argv[1])).message)' $starsDirection
            $starsExit = $LASTEXITCODE
            Start-Sleep -Milliseconds 250
            $starsAfter = [StarsVolumeProbe]::Read()
            $starsChanged = if ($starsDirection -eq 'volume_up') { $starsAfter.Level -gt $starsBefore.Level } else { $starsAfter.Level -lt $starsBefore.Level }
            $starsResults.tests += [ordered]@{intent=$starsDirection; before=$starsBefore.Level; after=$starsAfter.Level; changed_as_expected=$starsChanged; python_exit=$starsExit; response=($starsReply -join ' ')}
        }
    } finally {
        [StarsVolumeProbe]::Restore($starsInitialState.Level, $starsInitialState.Muted)
        $starsRestored = [StarsVolumeProbe]::Read()
        $starsResults.restored = [Math]::Abs($starsRestored.Level-$starsInitialState.Level) -lt 0.0001 -and $starsRestored.Muted -eq $starsInitialState.Muted
    }
}
$starsResults | ConvertTo-Json -Depth 5
