#
# Equations for the electrode-electrolyte interface, for lead-acid reactions
#
import pybamm
import autograd.numpy as np


class MainReaction(pybamm.interface.InterfacialCurrent, pybamm.LeadAcidBaseModel):
    """
    Interfacial current from lead-acid reactions

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`InterfacialCurrent`, :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_exchange_current_densities(self, c_e, domain=None):
        """The exchange current-density as a function of concentration

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case c_e.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density

        """
        param = self.set_of_parameters
        domain = domain or c_e.domain

        if domain == ["negative electrode"]:
            return param.m_n * c_e
        elif domain == ["positive electrode"]:
            c_w = param.c_w(c_e)
            return param.m_p * c_e ** 2 * c_w


class OxygenReaction(pybamm.interface.InterfacialCurrent, pybamm.LeadAcidBaseModel):
    """
    Interfacial current from oxygen reactions in a lead-acid battery

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.InterfacialCurrent`, :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_butler_volmer(self, j0a, j0c, eta_r, domain=None):
        """See :meth:`pybamm.interface.InterfacialCurrent.get_butler_volmer`"""
        param = self.set_of_parameters
        return (j0a * pybamm.Function(np.exp, eta_r / param.ne_Ox)) - (
            j0c * pybamm.Function(np.exp, -eta_r / param.ne_Ox)
        )

    def get_exchange_current_densities(self, c_e, c_o, domain=None):
        """The exchange current-density as a function of concentration

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        c_o : :class:`pybamm.Symbol`
            Oxygen concentration
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case c_e.domain is used.

        Returns
        -------
        tuple of :class:`pybamm.Symbol`
            Forward and reverse exchange-current densities (j0a, j0c)

        """
        raise NotImplementedError


class SrinivasanOxygenReaction(OxygenReaction):
    """
    Srinivasan form of oxygen reactions

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`OxygenReaction`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_exchange_current_densities(self, c_e, c_o, domain=None):
        """See :meth:`OxygenReaction.get_exchange_current_densities` """
        param = self.set_of_parameters
        domain = domain or c_e.domain

        if domain == ["negative electrode"]:
            j0_Ox_ref = param.j0_Ox_n_ref
        elif domain == ["positive electrode"]:
            j0_Ox_ref = param.j0_Ox_p_ref
        return (j0_Ox_ref * c_e, j0_Ox_ref * c_e * c_o / param.c_ox_ref)


class BernardiOxygenReaction(OxygenReaction):
    """
    Bernardi form of oxygen reactions

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`OxygenReaction`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_exchange_current_densities(self, c_e, c_o, domain=None):
        """See :meth:`OxygenReaction.get_exchange_current_densities` """
        param = self.set_of_parameters
        domain = domain or c_e.domain

        if domain == ["negative electrode"]:
            j0_Ox_ref = param.j0_Ox_n_ref
        elif domain == ["positive electrode"]:
            j0_Ox_ref = param.j0_Ox_p_ref
        return (j0_Ox_ref * c_e, j0_Ox_ref * c_e * p_o)
