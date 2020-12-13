import re

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

        # get number of arities and name of each predicate
        self._arity_num = {}
        self._predicates_name = {}
        for predicate in self.predicates:
            self._arity_num[predicate] = len(predicate.split(','))
            self._predicates_name[predicate] = predicate.split('(')[0]
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
            raise RuntimeError("Incompatible weights size")
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
    def formula_vars(self):
        total_vars = []
        for f in self.formulas:
            f_vars = set()
            for p, name in self._predicates_name.items():
                res = re.findall(r'{}\(([^\)]+)\)'.format(name), f)
                logger.debug('found arities %s of %s in %s', res, name, f)
                if len(res) > 0:
                    for r in res:
                        arities = r.split(',')
                        f_vars.update(arities)
            f_vars = list(filter(lambda x: not x.isnumeric(), f_vars))
            logger.debug(f_vars)
            total_vars.append(len(f_vars))
        return total_vars

    @property
    def world_size(self):
        if self.domain_size[0] == 0:
            return 0
        return 2 ** self.herbrand_size
