from logzero import logger

from mln import MLN
from approxWFOMC.logic import Constant


def convert2mln(clauses, domain_size):
    """
    Clauses example:
    [
        [-aux1(X), -stress(X), smokes(X)],
        [aux1(X), stress(X)],
        [aux1(X), -smokes(X)],

        [-aux2(X,Y), -friends(X,Y), -smokes(X), smokes(Y)],
        [aux2(X,Y), friends(X,Y)],
        [aux2(X,Y), smokes(X)],
        [aux2(X,Y), -smokes(Y)],

        [-aux3(X,Y,Z), -friends(X,Y), -friends(Y,Z), friends(X,Z)],
        [aux3(X,Y,Z), friends(X,Y)],
        [aux3(X,Y,Z), friends(Y,Z)],
        [aux3(X,Y,Z), -friends(X,Z)]
    ]
    """
    # for calculate upper and lower bound based on aux predicate weights
    aux2dim = {}
    # support multiple domains in the future
    domain_name = ['domain']
    predicates = set()
    formulas = []
    for c in clauses:
        # only consider [-aux_(_, ...), p1(_, ...), p2(_,...), ...]
        # i.e. aux_(_, ...) => p1(_, ...) v p2(_, ...) v ...
        aux_predicate = None
        has_aux = False
        for p in c:
            if p.name.startswith('aux'):
                has_aux = True
            if p.negated and p.is_aux():
                aux_predicate = p
                break
        # formula of mln in wmc must contain aux pred!
        if not has_aux:
            logger.debug('evidence literal: %s', c)
            continue
        if not aux_predicate:
            continue
        # formula with more than 2 variables is not supported by WFOMC
        if len(aux_predicate.free_variables()) > 2:
            continue
        formula = []
        for p in c:
            if p.is_aux():
                continue
            predicates.add('{}({})'.format(
                p.name, ','.join(domain_name * len(p.args)))
            )
            p_str = ''
            if p.negated:
                p_str += '!'
            args = []
            for a in p.args:
                if isinstance(a, Constant) and not a.name.isnumeric():
                    raise RuntimeError('grounding atom should have numeircal name')
                args.append(a.name.lower())
            p_str += '{}({})'.format(p.name, ','.join(args))
            formula.append(p_str)
        aux2dim[aux_predicate.name] = len(formulas)
        formulas.append(' v '.join(formula))
    logger.debug('predicates: %s\nformulas: %s', list(predicates), formulas)
    return MLN(domain_name, predicates, formulas, [domain_size]), aux2dim
