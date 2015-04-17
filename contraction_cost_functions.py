__author__ = 'kazjon'

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from theano import tensor

class ContractiveCost(DefaultDataSpecsMixin, Cost):
    """
    Contractive autoencoder cost.

    Shit I cobbled together to call the contraction functions.

    Parameters
    ----------
    coeff : float
        Coefficient for this regularization term in the objective
        function.
    """
    def __init__(self, coeff):
        self.coeff = coeff

    def expr(self, model, data, **kwargs):
        """
        Calculate regularization penalty.
        """
        penalty = self.coeff * model.contraction_penalty(data)
        penalty.name = 'contraction_penalty'
        return penalty


class HigherOrderContractiveCost(DefaultDataSpecsMixin, Cost):
    """
    Contractive autoencoder cost.

    Shit I cobbled together to call the contraction functions.

    Parameters
    ----------
    coeff : float
        Coefficient for this regularization term in the objective
        function.
    """
    def __init__(self, coeff):
        self.coeff = coeff

    def expr(self, model, data, **kwargs):
        """
        Calculate regularization penalty.
        """
        penalty = self.coeff * model.higher_order_penalty(data)
        penalty.name = 'higher_order_contraction_penalty'
        return penalty
