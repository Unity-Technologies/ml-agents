using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text.RegularExpressions;
using NUnit.Framework;

namespace Unity.MLAgents.Tests.Communicator
{
    /// <summary>
    /// Grpc.Core.dll embeds its own TLS root list (Grpc.Core.roots.pem), which ships in Standalone players.
    /// These tests make sure roots tied to the eMudhra intermediate CA compromise stay out of it.
    /// </summary>
    [TestFixture]
    public class GrpcRootCertificatesTests
    {
        const string k_RootsResourceName = "Grpc.Core.roots.pem";

        // SHA-256 fingerprints of removed roots.
        static readonly string[] k_RemovedRootFingerprints =
        {
            // AAA Certificate Services (Comodo CA Limited): cross-signed the compromised eMudhra intermediates.
            "D7A7A0FB5D7E2731D771E9484EBCDEF71D5F0C3E0A2948782BC83EE0EA699EF4",
        };

        // Subject names of roots that must not be present, including the eMudhra roots themselves.
        static readonly string[] k_RemovedRootNames =
        {
            "AAA Certificate Services",
            "emSign",
        };

        static List<X509Certificate2> LoadEmbeddedRoots()
        {
            var grpcAssembly = typeof(Grpc.Core.Channel).Assembly;
            using (var stream = grpcAssembly.GetManifestResourceStream(k_RootsResourceName))
            {
                Assert.IsNotNull(stream, $"{k_RootsResourceName} not found in {grpcAssembly.FullName}");
                var pem = new StreamReader(stream).ReadToEnd();
                return Regex.Matches(pem, "-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----", RegexOptions.Singleline)
                    .Cast<Match>()
                    .Select(m => new X509Certificate2(Convert.FromBase64String(Regex.Replace(m.Groups[1].Value, @"\s", ""))))
                    .ToList();
            }
        }

        static string Sha256Fingerprint(X509Certificate2 cert)
        {
            using (var sha = SHA256.Create())
            {
                return BitConverter.ToString(sha.ComputeHash(cert.RawData)).Replace("-", "");
            }
        }

        [Test]
        public void TestEmbeddedRootsAreParseable()
        {
            // Guards against a broken resource: the list should still hold the rest of the roots.
            Assert.Greater(LoadEmbeddedRoots().Count, 100);
        }

        [Test]
        public void TestEmbeddedRootsExcludeRemovedFingerprints()
        {
            var found = LoadEmbeddedRoots()
                .Where(c => k_RemovedRootFingerprints.Contains(Sha256Fingerprint(c)))
                .Select(c => c.Subject)
                .ToList();
            Assert.IsEmpty(found, "Removed roots found in " + k_RootsResourceName + ": " + string.Join("; ", found));
        }

        [Test]
        public void TestEmbeddedRootsExcludeRemovedNames()
        {
            var found = LoadEmbeddedRoots()
                .Select(c => c.Subject)
                .Where(s => k_RemovedRootNames.Any(n => s.IndexOf(n, StringComparison.OrdinalIgnoreCase) >= 0))
                .ToList();
            Assert.IsEmpty(found, "Removed roots found in " + k_RootsResourceName + ": " + string.Join("; ", found));
        }
    }
}
