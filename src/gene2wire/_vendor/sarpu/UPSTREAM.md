# SAR-PU upstream snapshot

- Repository: <https://github.com/ML-KULeuven/SAR-PU>
- Pinned repository commit: `6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7`
- Last upstream commit touching the vendored Python sources: `b75f652ba90b255e9edee9d978c419b230636155`
- License: MIT; the unmodified upstream license text is in `LICENSE`.
- Imported paths: `sarpu/sarpu/pu_learning.py` and
  `sarpu/sarpu/PUmodels.py`.

The statistical routine in `pu_learning.py` is used without source changes.
`PUmodels.py` has one packaging-only difference: this snapshot has a final LF
at end of file, while the upstream blob does not.  Its Python tokens and
behavior are otherwise byte-for-byte identical.  Compatibility with current
scikit-learn releases and Gene2Wire's per-target measurement mask is provided
in a separate wrapper module; it is not represented as an upstream change.

| File | Upstream SHA-256 | Vendored SHA-256 | Upstream git blob |
|---|---|---|---|
| `pu_learning.py` | `6d929a4db0555b2d2e1ef05686618d819decc0bdfa3bbdcab59fec8ba4261b43` | `6d929a4db0555b2d2e1ef05686618d819decc0bdfa3bbdcab59fec8ba4261b43` | `f218fe5ffd6a4d1615b06cd78cfc19d957963eac` |
| `PUmodels.py` | `28b41fd8adb50231c500b937de2ef924c9bc6fb0e597d574b1a6b1c91035403a` | `bbef08af6c9b7dae2536dc07602cfdf355757ca6874cc1e66a55b5d890547cb4` | `6241b21491436549a86dafe60510db0184d64bf3` |
| `LICENSE` | `3ffe98aa8f5153033951f5cbc2322c8c8943704827dcfefd188eea72b5dec52e` | `3ffe98aa8f5153033951f5cbc2322c8c8943704827dcfefd188eea72b5dec52e` | `87bd8db562a5849d3ca7f19f7782b1bc9ae1785a` |

The hashes are checked by the Gene2Wire wrapper before any fit.  A source
mismatch fails closed rather than silently using a modified implementation.
