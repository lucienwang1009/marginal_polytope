from logzero import logger


class MLN(object):
    def __init__(self, domain_name, predicates, formulas,
                 domain_size, formula_weights=None):
        if len(domain_name) > 1:
            raise RuntimeError("Not support multiple domains")
        self.domain_name = domain_name
        self.predicates = predicates
        self.formulas = formulas
        self.domain_size = domain_size
        self._formula_weights = formula_weights
        self.sanity_check()

        # get number of arities in each predicate
        self._arity_num = {}
        for predicate in self.predicates:
            self._arity_num[predicate] = len(predicate.split(','))
        logger.debug('arity num: %s', self._arity_num)

    def sanity_check(self):
        if self.formula_weights is not None and \
                len(self.formula_weights) != len(self.formulas):
            raise RuntimeError("Incompatible weight size")
        if len(self.domain_name) != len(self.domain_size):
            raise RuntimeError("Incompatible domain number")

    @property
    def formula_weights(self):
        return self._formula_weights

    @formula_weights.setter
    def formula_weights(self, weights):
        if len(weights) != len(self.formulas):
            raise RuntimeError("Incompatible weight size")
        self._formula_weights = weights

    @property
    def herbrand_size(self):
        if self.domain_size[0] == 0:
            return 0
        total = 0
        for n in self._arity_num.values():
            total += (self.domain_size[0] ** n)
        logger.debug('herbrand size: %s', total)
        return total

    @property
    def world_size(self):
        if self.domain_size[0] == 0:
            return 0
        return 2 ** self.herbrand_size
