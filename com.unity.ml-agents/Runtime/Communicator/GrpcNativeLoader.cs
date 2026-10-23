#if UNITY_EDITOR || UNITY_STANDALONE
using System;
using System.Reflection;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Unity.MLAgents
{
    /// <summary>
    /// Works around Grpc.Core 1.10 failing to load its native extension on Linux under CoreCLR.
    ///
    /// Grpc.Core only special-cases Mono and ".NET Core"; on any other runtime (e.g. Unity's CoreCLR)
    /// it P/Invokes dlopen/dlsym from "libdl.so". That unversioned name only exists when glibc dev
    /// packages are installed, so loading fails on most machines (only "libdl.so.2" is present).
    /// We hook the load context's ResolvingUnmanagedDll event, which is only raised after the default
    /// lookup fails, and redirect "libdl.so" to "libdl.so.2" (or libc, which exports dlopen on glibc 2.34+).
    ///
    /// Everything is done via reflection so this compiles against all API profiles and is a no-op on
    /// Mono and IL2CPP, where these types don't exist.
    /// </summary>
    internal static class GrpcNativeLoader
    {
        static readonly string[] k_LibdlCandidates = { "libdl.so.2", "libc.so.6" };

        static bool s_Installed;
        static MethodInfo s_TryLoad;

        internal static void EnsureInstalled()
        {
            if (s_Installed)
            {
                return;
            }
            s_Installed = true;

            if (Application.platform != RuntimePlatform.LinuxPlayer &&
                Application.platform != RuntimePlatform.LinuxEditor &&
                Application.platform != RuntimePlatform.LinuxServer)
            {
                return;
            }

            try
            {
                var coreLib = typeof(Marshal).Assembly;
                var nativeLibrary = coreLib.GetType("System.Runtime.InteropServices.NativeLibrary");
                var loadContextType = coreLib.GetType("System.Runtime.Loader.AssemblyLoadContext");
                if (nativeLibrary == null || loadContextType == null)
                {
                    // Not running on CoreCLR (Mono / IL2CPP): Grpc.Core handles these itself.
                    return;
                }

                s_TryLoad = nativeLibrary.GetMethod(
                    "TryLoad", new[] { typeof(string), typeof(IntPtr).MakeByRefType() });
                var getLoadContext = loadContextType.GetMethod(
                    "GetLoadContext", BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(Assembly) }, null);
                var resolvingEvent = loadContextType.GetEvent("ResolvingUnmanagedDll");
                if (s_TryLoad == null || getLoadContext == null || resolvingEvent == null)
                {
                    return;
                }

                var grpcAssembly = typeof(Grpc.Core.Channel).Assembly;
                var loadContext = getLoadContext.Invoke(null, new object[] { grpcAssembly });
                if (loadContext == null)
                {
                    return;
                }

                Func<Assembly, string, IntPtr> handler = Resolve;
                resolvingEvent.AddEventHandler(loadContext, handler);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ML-Agents: unable to install gRPC native library fallback: {e}");
            }
        }

        static IntPtr Resolve(Assembly assembly, string libraryName)
        {
            if (libraryName != "libdl.so" || s_TryLoad == null)
            {
                return IntPtr.Zero;
            }

            foreach (var candidate in k_LibdlCandidates)
            {
                var args = new object[] { candidate, IntPtr.Zero };
                if ((bool)s_TryLoad.Invoke(null, args))
                {
                    return (IntPtr)args[1];
                }
            }
            return IntPtr.Zero;
        }
    }
}
#endif
