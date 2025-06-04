import jax.numpy as jnp
from smefit import loader
from smefit import compute_theory as pr
from smefit.rge.rge import load_rge_matrix

class CMS_DYMee_13TeV:
    def __init__(self, coefficients, rge_dict=None, **kwargs):

        operators = {
            k: {"max": 0.0, "min": 0.0} for k in coefficients.name
        }

        if rge_dict is not None:
            rgematrix, operators_to_keep = load_rge_matrix(
                rge_dict=rge_dict,
                coeff_list=list(operators.keys()),
                datasets=[{'name': 'CMS_DYMee_13TeV', 'order': kwargs['order']}],
                theory_path='./theory',
                cutoff_scale=None,
                result_path='./results',
                result_ID='CMS_DYMee_13TeV',
            )
        else:
            operators_to_keep=operators
            rgematrix=None

        self.drell_yan_dataset = loader.load_datasets(
            commondata_path='./commondata',
            datasets=[{'name': 'CMS_DYMee_13TeV', 'order': kwargs['order']}],
            operators_to_keep=operators_to_keep,
            use_quad=kwargs['use_quad'],
            use_theory_covmat=True,
            use_t0=False,
            use_multiplicative_prescription=False,
            default_order=kwargs['order'],
            theory_path='./theory',
            rot_to_fit_basis=None,
            has_uv_couplings=False,
            has_external_chi2=True,
            rgemat=rgematrix,
            cutoff_scale=None,
        )
        self.use_quad = kwargs['use_quad']

    def compute_chi2(self, coefficient_values):
        # Compute theory predictions
        theory = pr.make_predictions(
            self.drell_yan_dataset, coefficient_values, self.use_quad, False
        )

        theory = jnp.where(theory > 0, theory, 1e-6)
        data = self.drell_yan_dataset.Commondata

        # If the measured number of events is 0, manually put x*log(x)=0
        log_term = jnp.where(data > 1e-6, data * jnp.log(data / theory), 0.0)

        chi2 = 2. * jnp.sum(theory - data + log_term)
        return chi2
