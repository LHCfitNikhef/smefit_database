# python
import pathlib
import numpy as np
import pytest
from smefit.loader import Loader
from smefit.covmat import covmat_from_systematics

COMMONDATA_DIRS = ["../commondata", "../commondata_projections_L0"]


def get_yaml_files(directories):
    files = []
    for directory in directories:
        if pathlib.Path(directory).is_dir():
            files.extend(
                (f.stem, directory)
                for f in pathlib.Path(directory).iterdir()
                if f.suffix == ".yaml"
            )
    return files


@pytest.mark.parametrize("dataset,commondata_dir", get_yaml_files(COMMONDATA_DIRS))
def test_experimental_covmat(dataset, commondata_dir):
    Loader.commondata_path = pathlib.Path(commondata_dir)
    Loader.theory_path = pathlib.Path("../theory")

    loaded_dataset = Loader(
        setname=dataset,
        operators_to_keep=[],
        order="LO",
        use_quad="False",
        use_theory_covmat="False",
        use_multiplicative_prescription="False",
        rot_to_fit_basis=None,
        cutoff_scale=None,
    )

    stat_error = loaded_dataset.stat_error
    sys_error = loaded_dataset.sys_error

    exp_covmat = covmat_from_systematics([stat_error], [sys_error])

    eigvals = np.linalg.eigvalsh(exp_covmat)
    assert np.all(
        eigvals > 0
    ), f"Experimental covariance matrix for {dataset} is not positive definite."
