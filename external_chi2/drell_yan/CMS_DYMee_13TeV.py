import jax.numpy as jnp
from smefit import loader
from smefit import compute_theory as pr
from smefit.rge.rge import load_rge_matrix
import os


class CMS_DYMee_13TeV:
    def __init__(
        self,
        coefficients,
        rge_dict=None,
        use_quad=True,
        order="LO",
        use_theory_covmat=True,
        use_t0=True,
        use_multiplicative_prescription=False,
        cutoff_scale=None,
        save_rge_path=None,
        rg_matrix=None,
    ):
        """
        Initialize the CMS_DYMee_13TeV class.

        coefficients: The Wilson coefficients to be used in the analysis.
        rge_dict: A dictionary containing the RGE information.
        use_quad: Whether to add the quadratic contributions.
        order: The order of the calculation (e.g., "LO", "NLO_QCD").
        use_theory_covmat: Whether to use the theory covariance matrix.
        use_t0: Whether to use the t0 prescription.
        use_multiplicative_prescription: Whether to use the multiplicative prescription.
        cutoff_scale: The cutoff scale for the data kinematics.
        save_rge_path: The path to save the RGE matrix.
        rg_matrix: A pre-computed RGE matrix.
        """

        operators = {k: {"max": 0.0, "min": 0.0} for k in coefficients.name}
        theory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theory")
        commondata_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "commondata"
        )
        if save_rge_path is not None:
            os.makedirs(os.path.join(save_rge_path, "CMS_DYMee_13TeV"), exist_ok=True)

        # If a pre-computed rge matrix is provided, add it to the rge_dict
        # If not, set it to False in case it was defined for the datasets
        if rg_matrix is not None:
            rge_dict["rg_matrix"] = rg_matrix
        else:
            rge_dict["rg_matrix"] = False

        if rge_dict is not None:
            rgematrix, operators_to_keep = load_rge_matrix(
                rge_dict=rge_dict,
                coeff_list=list(operators.keys()),
                datasets=[{"name": "CMS_DYMee_13TeV", "order": order}],
                theory_path=theory_path,
                cutoff_scale=cutoff_scale,
                result_path=save_rge_path,
                result_ID="CMS_DYMee_13TeV",
            )
        else:
            operators_to_keep = operators
            rgematrix = None

        self.drell_yan_dataset = loader.load_datasets(
            commondata_path=commondata_path,
            datasets=[{"name": "CMS_DYMee_13TeV", "order": order}],
            operators_to_keep=operators_to_keep,
            use_quad=use_quad,
            use_theory_covmat=use_theory_covmat,
            use_t0=use_t0,
            use_multiplicative_prescription=use_multiplicative_prescription,
            default_order=order,
            theory_path=theory_path,
            rgemat=rgematrix,
            cutoff_scale=cutoff_scale,
        )
        self.use_multiplicative_prescription = use_multiplicative_prescription
        self.use_quad = use_quad

    def compute_chi2(self, coefficient_values):
        # Compute theory predictions
        theory = pr.make_predictions(
            dataset=self.drell_yan_dataset,
            coefficients_values=coefficient_values,
            use_quad=self.use_quad,
            use_multiplicative_prescription=self.use_multiplicative_prescription,
        )

        theory = jnp.where(theory > 0, theory, 1e-6)
        data = self.drell_yan_dataset.Commondata

        # If the measured number of events is 0, manually put x*log(x)=0
        log_term = jnp.where(data > 1e-6, data * jnp.log(data / theory), 0.0)

        chi2 = 2.0 * jnp.sum(theory - data + log_term)
        return chi2
