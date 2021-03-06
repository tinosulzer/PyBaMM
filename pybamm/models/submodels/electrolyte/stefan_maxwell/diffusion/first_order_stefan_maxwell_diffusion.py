#
# Class for electrolyte diffusion employing stefan-maxwell (first-order)
#
import pybamm
from .base_stefan_maxwell_diffusion import BaseModel


class FirstOrder(BaseModel):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (First-order refers to first-order term in
    asymptotic expansion)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.diffusion.BaseModel`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions)

    def get_coupled_variables(self, variables):
        param = self.param
        l_n = param.l_n
        l_s = param.l_s
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack
        T_0 = variables["Leading-order cell temperature"]
        c_e_0 = variables["Leading-order x-averaged electrolyte concentration"]
        # v_box_0 = variables["Leading-order volume-averaged velocity"]
        dc_e_0_dt = variables["Leading-order electrolyte concentration change"]
        eps_n_0 = variables["Leading-order x-averaged negative electrode porosity"]
        eps_s_0 = variables["Leading-order x-averaged separator porosity"]
        eps_p_0 = variables["Leading-order x-averaged positive electrode porosity"]
        deps_n_0_dt = variables[
            "Leading-order x-averaged negative electrode porosity change"
        ]
        deps_p_0_dt = variables[
            "Leading-order x-averaged positive electrode porosity change"
        ]

        # Combined time derivatives
        d_epsc_n_0_dt = c_e_0 * deps_n_0_dt + eps_n_0 * dc_e_0_dt
        d_epsc_s_0_dt = eps_s_0 * dc_e_0_dt
        d_epsc_p_0_dt = c_e_0 * deps_p_0_dt + eps_p_0 * dc_e_0_dt

        # Right-hand sides
        rhs_n = d_epsc_n_0_dt - sum(
            reaction["Negative"]["s"]
            * variables[
                "Leading-order x-averaged " + reaction["Negative"]["aj"].lower()
            ]
            for reaction in self.reactions.values()
        )
        rhs_s = d_epsc_s_0_dt
        rhs_p = d_epsc_p_0_dt - sum(
            reaction["Positive"]["s"]
            * variables[
                "Leading-order x-averaged " + reaction["Positive"]["aj"].lower()
            ]
            for reaction in self.reactions.values()
        )

        # Diffusivities
        D_e_n = (eps_n_0 ** param.b_n) * param.D_e(c_e_0, T_0)
        D_e_s = (eps_s_0 ** param.b_s) * param.D_e(c_e_0, T_0)
        D_e_p = (eps_p_0 ** param.b_p) * param.D_e(c_e_0, T_0)

        # Fluxes
        N_e_n_1 = -pybamm.outer(rhs_n, x_n)
        N_e_s_1 = -(
            pybamm.outer(rhs_s, (x_s - l_n))
            + pybamm.PrimaryBroadcast(rhs_n * l_n, "separator")
        )
        N_e_p_1 = -pybamm.outer(rhs_p, (x_p - 1))

        # Concentrations
        c_e_n_1 = pybamm.outer(rhs_n / (2 * D_e_n), x_n ** 2 - l_n ** 2)
        c_e_s_1 = pybamm.outer(rhs_s / 2, (x_s - l_n) ** 2) + pybamm.outer(
            rhs_n * l_n / D_e_s, x_s - l_n
        )
        c_e_p_1 = pybamm.outer(
            rhs_p / (2 * D_e_p), (x_p - 1) ** 2 - l_p ** 2
        ) + pybamm.PrimaryBroadcast(
            (rhs_s * l_s ** 2 / (2 * D_e_s)) + (rhs_n * l_n * l_s / D_e_s),
            "positive electrode",
        )

        # Correct for integral
        c_e_n_1_av = -rhs_n * l_n ** 3 / (3 * D_e_n)
        c_e_s_1_av = (rhs_s * l_s ** 3 / 6 + rhs_n * l_n * l_s ** 2 / 2) / D_e_s
        c_e_p_1_av = (
            -rhs_p * l_p ** 3 / (3 * D_e_p)
            + (rhs_s * l_s ** 2 * l_p / (2 * D_e_s))
            + (rhs_n * l_n * l_s * l_p / D_e_s)
        )
        A_e = -(eps_n_0 * c_e_n_1_av + eps_s_0 * c_e_s_1_av + eps_p_0 * c_e_p_1_av) / (
            l_n * eps_n_0 + l_s * eps_s_0 + l_p * eps_p_0
        )
        c_e_n_1 += pybamm.PrimaryBroadcast(A_e, "negative electrode")
        c_e_s_1 += pybamm.PrimaryBroadcast(A_e, "separator")
        c_e_p_1 += pybamm.PrimaryBroadcast(A_e, "positive electrode")
        c_e_n_1_av += A_e
        c_e_s_1_av += A_e
        c_e_p_1_av += A_e

        # Update variables
        c_e = pybamm.Concatenation(
            pybamm.PrimaryBroadcast(c_e_0, "negative electrode") + param.C_e * c_e_n_1,
            pybamm.PrimaryBroadcast(c_e_0, "separator") + param.C_e * c_e_s_1,
            pybamm.PrimaryBroadcast(c_e_0, "positive electrode") + param.C_e * c_e_p_1,
        )
        variables.update(self._get_standard_concentration_variables(c_e))
        # Update with analytical expressions for first-order x-averages
        variables.update(
            {
                "X-averaged first-order negative electrolyte concentration": c_e_n_1_av,
                "X-averaged first-order separator concentration": c_e_s_1_av,
                "X-averaged first-order positive electrolyte concentration": c_e_p_1_av,
            }
        )

        N_e = pybamm.Concatenation(
            param.C_e * N_e_n_1, param.C_e * N_e_s_1, param.C_e * N_e_p_1
        )
        variables.update(self._get_standard_flux_variables(N_e))

        return variables
