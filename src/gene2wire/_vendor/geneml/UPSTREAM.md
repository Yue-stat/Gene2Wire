# GenEML upstream snapshot

- Repository: <https://github.com/nirbhayjm/GenEML>
- Pinned commit: `f6c08c8f3c69a009b565955231509633eda611d1`
- License: MIT; the unmodified license text is in `LICENSE`.
- Imported paths: root `README.md`, and `src/model.py`, `src/ops.py`,
  `src/inputs.py`, and `src/main.py`.

The upstream implementation targets Python 2.7.  The original sources are
stored with a `.py2` suffix so packaging or test discovery cannot import them
as Python 3 modules.  Four files and the license are byte-for-byte identical
to the pinned Git blobs.  `ops.py2` has one packaging-only difference: this
snapshot adds a final LF, which is absent from the upstream blob.  Its Python
tokens and behavior are otherwise identical.

The runnable port lives in `gene2wire.experiments.geneml_authors`.  It keeps
the authors' inductive factorization, Pólya–Gamma updates, stochastic
sufficient statistics, and one global exposure probability per target.  Its
documented adaptations are limited to Python 3/array compatibility and a
measurement mask: an entry with `measured=False` contributes to no latent,
target-factor, or exposure sufficient statistic.  Rows with no measured
targets are also excluded from the feature-map statistic.  With an all-true
mask, the masked entry point calls exactly the same numerical port as its
unmasked entry point; this reduction is regression-tested.

There is not yet a Python-2-vs-Python-3 golden-output fixture from an original
published dataset.  Consequently Gene2Wire must describe this as an
**authors-source Python 3 measurement-mask port**, not as an unmodified run of
the authors' Python 2 program.

| File | Vendored git blob | Upstream git blob |
|---|---|---|
| `LICENSE` | `775c1cc73d9108424ffa5ee6471ef2a6a5643a72` | same |
| `README.upstream.md` | `e727de456d44b0dfd5c5af6544925f59672e13a3` | same |
| `inputs.py2` | `398b75da21cd3a15e69b6d16bc39b164e34c2aeb` | same |
| `main.py2` | `7962bad437dff0354fcbaf38f400d5131b5a7977` | same |
| `model.py2` | `637848adc3ed4a4bc4e968504b849429e1f6db25` | same |
| `ops.py2` | `8228238d7e7aafe6d7cbbf5b6e573088fcb6f839` | `785eaaf565840c34dee73e425ea9375a9eb80b35` (final LF only) |

