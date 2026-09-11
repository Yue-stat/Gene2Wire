# Python source layout

The installable Gene2Wire package is in [`gene2wire/`](gene2wire/). This is the
standard Python `src` layout: install the project from the repository root and
import it as `gene2wire`.

Reusable estimators and tuning code live directly in the package. Dataset
adapters and shared experiment orchestration live in `gene2wire/experiments/`;
the runnable notebook entry points are kept separately under
[`notebooks/`](../notebooks/).
