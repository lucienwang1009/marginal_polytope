import argparse
import logzero
import logging
import re
import heapq
import itertools
import math
import tempfile
import subprocess
import time
import numpy as np

from math import floor
from contexttimer import Timer
from pyparsing import Word, alphas, nums, alphanums, \
    Suppress, OneOrMore, Group, Optional, delimitedList
from copy import deepcopy
from logzero import logger
from scipy.optimize import linprog

from approxWFOMC.converter import convert2mln
from approxWFOMC.logic import Predicate, Variable, Constant
from main import main as get_irmp
from utils import plot_convex_hull

varcount = 0


class DimacsClause:
    def __init__(self, *args):
        self.args = sorted(args)

    def __hash__(self):
        return hash(str(self.args))

    def __eq__(self, other):
        return other and self.args == other.args

    def __repr__(self):
        return " ".join(map(str, self.args)) + " 0"


def is_var(x):
    return x.istitle()


def get_sharpsat_model_count(input_str):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
        tf.write(input_str)
        tf.flush()
        out = subprocess.check_output(['./sharpSAT', tf.name],
                                      universal_newlines=True)
        return int(out.split('\n')[-6])


def get_approxmc_model_count(input_str, i, mmax):
    if i is not None:
        delta = 0.2 / (i * math.log(mmax + 1))
    else:
        delta = 0.2
    logger.info("Delta value for approxmc call %s is %s", i, str(delta))
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
        tf.write(input_str)
        tf.flush()
        out = subprocess.run(['./approxmc', '--delta', str(delta), tf.name], check=False,
                             stdout=subprocess.PIPE,
                             universal_newlines=True).stdout
        cell, has = re.findall(r'is: (\d+) x 2\^(\d+)', out.split('\n')[-2])[0]
        return int(cell) * math.pow(2, int(has))

# Toggle ApproxMC/sharpSAT usage in the line below
get_model_count = lambda x, i, m: get_approxmc_model_count(x, i, m)
heuristic = lambda ub, lb: ub - lb
logheuristic = lambda ub, lb: math.log(ub) - math.log(lb)

def parseInput(input):
    variable = Word(alphas.upper(), alphanums)
    predicate_name = Word(alphanums + '_' + '-')
    ground_atom = Word(nums)
    decimal_number = Word(nums + '.' + '-')
    # negation = Literal('-')

    # nonground_predicate = predicate_name + Suppress('(') + Group(Optional(delimitedList(variable))) + Suppress(')')
    predicate = predicate_name + Suppress('(') + Group(Optional(delimitedList(variable ^ ground_atom))) + Suppress(')')

    predicates = OneOrMore(Group(Suppress("predicate") + predicate_name + Word(nums) + decimal_number + decimal_number))
    clause = delimitedList(Group(predicate))
    clauses = OneOrMore(Group(clause))
    spec = Group(predicates) + Group(clauses)
    return spec.parseString(input)


def main():
    parser = argparse.ArgumentParser(description='Perform approximate WFOMC on an input FO CNF.')
    parser.add_argument('domain', metavar='N', type=int, help='domain size')
    parser.add_argument('file', help='filename of input FO CNF')
    parser.add_argument('--improve', action='store_true', help='if use improved method to'
                        ' comput lower and upper bound')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--log', type=str, default='./log.txt')

    args = parser.parse_args()
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile(args.log, mode='w')

    with open(args.file, 'r') as file:
        data = file.read()
    domainsize = args.domain

    d = parseInput(data)

    arities = {}
    clauses = []
    for clause in d[1]:
        c = []
        for predicate in clause:
            predicate_args = []
            if predicate[0][0] == '-':
                predicatename = predicate[0][1:]
                negated = True
            else:
                predicatename = predicate[0]
                negated = False
            for pos, var in enumerate(predicate[1]):
                # capital is Variable
                if is_var(var):
                    predicate_args.append(Variable(var))
                else:
                    predicate_args.append(Constant(var))
            p = Predicate(predicatename, predicate_args, negated)
            arities[predicatename] = len(predicate_args)
            c.append(p)
        clauses.append(c)

    ######################################################
    # NOTE: start computing IRMP
    mln = None
    aux2dim = None
    if args.improve:
        mln, aux2dim = convert2mln(clauses, domainsize)
        logger.debug('mln:%s\naux2dim:%s', mln, aux2dim)
        convex_hull = get_irmp(mln, 'iter')
        plot_convex_hull(convex_hull, './polytope.png')
    ######################################################

    # At this point, clauses contains a list of lists of predicates
    # Start grounding
    sig = {}
    weights = {}
    for decl in d[0]:
        sig[decl[0]] = int(decl[1])
        weights[decl[0]] = (float(decl[2]), float(decl[3]))
    free_variables = list(get_free_variables(clauses))
    subs = itertools.product(map(lambda l: Constant(str(l)), range(domainsize)), repeat=len(free_variables))
    ground_clauses = []
    for sub in subs:
        acs = {}
        for q, var in enumerate(free_variables):
            acs[Variable(var.name)] = sub[q]
        ground_clauses += [list(map(lambda pr: apply_sub(pr, acs), clause)) for clause in clauses]

    # Finish grounding (see ground_clauses)

    # Start converting ground clauses to DIMACS representation
    index = {}
    out = set()
    for ground_clause in ground_clauses:
        outln = []
        for predicate in ground_clause:
            if predicate not in index:
                index[predicate] = fresh_variable()
            if (predicate.negated):
                outln.append(-1 * index[predicate])
            else:
                outln.append(index[predicate])
        out.add(DimacsClause(*outln))
    # Finish converting ground clauses to DIMACS representation (see out)

    # This maps predicates to their corresponding ground variables
    m = {}
    for k, v in index.items():
        if k.name not in m:
            m[k.name] = [v]
        else:
            m[k.name] = m[k.name] + [v]
    # Finish populating the map (see m)

    ############################################
    mmax = 1
    # for predicatename, arity in arities.items():
    #     mmax = mmax*(domainsize**arity + 1)
    # logger.info(mmax)
    # sys.exit(0)
    sampling_set = []
    non_aux_predicates = []
    aux_predicates = []
    for pred in weights.keys():
        # Use the non-auxiliary predicates as our sampling set
        if not pred.startswith("aux"):
            non_aux_predicates.append(pred)
            sampling_set += m[pred]
        # We only need to consider the auxiliary predicates when computing weights
        else:
            aux_predicates.append(pred)
            mmax = mmax * (domainsize ** arities[pred] + 1)

    logger.info("mmax value: %s", mmax)
    # logger.info(output_to_dimacs(out, sampling_set))
    start = time.time()
    number_of_mc_calls = 1
    non_heuristic_calls = 1
    mc = get_model_count(output_to_dimacs(out, sampling_set), non_heuristic_calls, mmax)
    logger.info("Initial model count is: %s", mc)

    tmin = 1
    tmax = 1
    priority_queue = []
    bounds = {}
    tolerance = 1.5
    # tolerance = 1
    # varcount keeps track of the highest variable ID in the DIMACS CNF,
    # so we know where to start building the constraints
    global varcount

    for pred in aux_predicates:
        # for pred in weights.keys():
        nog = domainsize ** arities[pred]
        bounds[pred] = (0, nog)

    if args.improve:
        tmin, tmax = get_upper_lower_bound_imp(convex_hull, bounds, weights, aux2dim,
                                               domainsize, arities)
    else:
        tmin, tmax = get_upper_lower_bound(bounds, weights, domainsize, arities)
    logger.debug('tmin: %s, tmax: %s', tmin, tmax)

    currentLb = mc * tmin
    currentUb = mc * tmax
    initialLb = currentLb
    initialUb = currentUb
    heapq.heappush(priority_queue, (-heuristic(currentUb, currentLb), id(bounds), mc, currentLb, currentUb, bounds))

    while True:
        if not (currentUb / currentLb > tolerance):
            logger.info("Converged!")
            break
        if not priority_queue:
            logger.info("Queue empty, terminating!")
            break
        logger.info("Current queue: %s", priority_queue)
        (_, _, parentMc, parentLb, parentUb, bounds) = heapq.heappop(priority_queue)
        logger.info("Popping the first (left-most) item off the queue: [%s, %s]", parentLb, parentUb)

        # tbounds = bounds
        temp = {}
        allsame = True
        best_pred = None
        best_pred_bounds = None
        for pred in aux_predicates:
            themin, themax = bounds[pred]
            if themin == themax:
                continue
            else:
                lr = [(themin, floor((themin + themax) / 2)), (floor((themin + themax) / 2) + 1, themax)]
                allsame = False

            lowb = []
            upb = []
            boundsb = []
            mcs = []
            for uu in lr:
                newbounds = deepcopy(bounds)
                newbounds[pred] = uu

                tmin = 1
                tmax = 1
                logger.info("Setting constraints: %s", newbounds)
                boundsb.append(newbounds)
                if args.improve:
                    # improvement 1: if the current bound isn't intersected with convex hull,
                    # stop split it!
                    bounds_to_check = [[]] * len(aux2dim)
                    for p_aux, dim in aux2dim.items():
                        bounds_to_check[dim] = newbounds[p_aux]
                    intersected = False
                    for point in itertools.product(*bounds_to_check):
                        if contains(convex_hull, point):
                            intersected = True
                            break
                    if not intersected:
                        logger.debug('New bound is not intersected with convex hull, skip it!')
                        mcs.append(0)
                        upb.append(0)
                        lowb.append(0)
                        continue

                    # improvement 2: calculate upper and lower bound with additional
                    # convex hull constrains
                    with Timer() as t:
                        tmin, tmax = get_upper_lower_bound_imp(
                            convex_hull, newbounds, weights, aux2dim,
                            domainsize, arities
                        )
                    logger.debug('elapsed time for calculating upper and lower bound: %s',
                                 t.elapsed)
                else:
                    tmin, tmax = get_upper_lower_bound(
                        newbounds, weights, domainsize, arities
                    )
                # NOTE: disable cache
                if len(mcs) > 0:
                    # if(parentMc < mcs[0]): # catch weird negative cases
                    #     mcA = 1
                    # else:
                    mcA = parentMc - mcs[0]
                    logger.info("Using cache to infer model count value for constraints above of: %s", mcA)
                else:
                    tout = deepcopy(out)
                    oldvc = varcount  # Cache the varcount before adding the constraints
                    # for pred in weights.keys():
                    for predic in aux_predicates:
                        amin, amax = newbounds[predic]
                        for x in encode_cardinality_constraint(amin, amax, m[predic]):
                            tout.add(DimacsClause(*list(x)))

                    number_of_mc_calls += 1
                    # ivalue += 1
                    mcA = get_model_count(output_to_dimacs(tout, sampling_set), None, mmax)
                    varcount = oldvc  # Restore the old varcount
                    logger.info("Model count for constraints above is: %s", mcA)
                # if mcA == 0:
                #     continue
                mcs.append(mcA)
                lowb.append(mcA * tmin)
                upb.append(mcA * tmax)
            logger.info("Left split bounds: [" + str(lowb[0]) + ", " + str(upb[0]) + "]")
            logger.info("Right split bounds: [" + str(lowb[1]) + ", " + str(upb[1]) + "]")
            testingub = upb[0] + upb[1]
            testinglb = lowb[0] + lowb[1]
            if testinglb <= 0 or testingub <= 0:  # catch weird negative case
                testinglb = initialLb
                testingub = initialUb
            if best_pred_bounds is None or \
                    logheuristic(best_pred_bounds[1], best_pred_bounds[0]) > logheuristic(testingub, testinglb):
                best_pred = pred
                best_pred_bounds = (testinglb, testingub)

            # logger.info("Log heuristic for current best bounds: %s", logheuristic(best_pred_bounds[1], best_pred_bounds[0]))
            # logger.info("Log heuristic for split above: %s", logheuristic(testingub, testinglb))
            temp[pred] = [(-heuristic(upb[0], lowb[0]), id(boundsb[0]), mcs[0], lowb[0], upb[0], boundsb[0]),
                          (-heuristic(upb[1], lowb[1]), id(boundsb[1]), mcs[1], lowb[1], upb[1], boundsb[1])]
        if allsame:
            break
        logger.info("Best predicate above was: %s", best_pred)
        currentUb = currentUb - parentUb
        currentLb = currentLb - parentLb
        currentUb += best_pred_bounds[1]
        currentLb += best_pred_bounds[0]
        #############################
        # NOTE: disable cache
        # Get a more accurate value for the right split of the selected predicate
        tout = deepcopy(out)
        oldvc = varcount  # Cache the varcount before adding the constraints
        # for pred in weights.keys():
        for predic in aux_predicates:
            amin, amax = temp[best_pred][1][5][predic]
            for x in encode_cardinality_constraint(amin, amax, m[predic]):
                tout.add(DimacsClause(*list(x)))

        number_of_mc_calls += 1  # if one of the splits is zero, we don't need to increase this
        non_heuristic_calls += 1
        updatedmc = get_model_count(output_to_dimacs(tout, sampling_set), non_heuristic_calls, mmax)
        varcount = oldvc  # Restore the old varcount
        logger.info("Got exact model count for right half of the split of the predicate selected: %s", updatedmc)

        currentLb -= best_pred_bounds[0]
        currentUb -= best_pred_bounds[1]
        mcs = [parentMc - updatedmc, updatedmc]
        for i, _ in enumerate(temp[best_pred]):
            if temp[best_pred][i][2] == 0:
                exact_lowb = 0
                exact_upb = 0
            else:
                exact_lowb = mcs[i] * temp[best_pred][i][3] * mcs[i] / temp[best_pred][i][2]
                exact_upb = mcs[i] * temp[best_pred][i][4] * mcs[i] / temp[best_pred][i][2]
            temp[best_pred][i] = (-heuristic(exact_upb, exact_lowb), temp[best_pred][i][1],
                                  mcs[i], exact_lowb, exact_upb, temp[best_pred][i][-1])
            currentLb += exact_lowb
            currentUb += exact_upb
        #############################
        for i in temp[best_pred]:
            heapq.heappush(priority_queue, i)
        logger.info("Current bounds on WMC: [" + str(currentLb) + ", " + str(currentUb) + "]")
        # logger.info("Number of model counter calls so far: %s", number_of_mc_calls)
        # logger.info("Number of non-heuristic model counter calls so far: %s", non_heuristic_calls)
        logger.info("==============")

    logger.info("Total number of model counter calls: %s", number_of_mc_calls)
    logger.info("Number of non-heuristic model counter calls so far: %s", non_heuristic_calls)
    logger.info("Best WMC bounds: [" + str(currentLb) + ", " + str(currentUb) + "]")
    end = time.time()
    logger.info("Runtime: %s", end - start)


def output_to_dimacs(clauses, sampling_set=None):
    p = "p cnf " + str(varcount) + " " + str(len(clauses)) + "\n"
    if sampling_set is not None:
        p += "c ind " + " ".join(map(str, sampling_set)) + " 0\n"
    p += '\n'.join(map(str, clauses))
    return p


def encode_cardinality_constraint(l, u, vars):
    clauses, finals = build_totalizer(vars)
    comparator = build_comparator(l, u, finals)
    clauses += comparator
    # Now do some primitive propagation on clauses before returning them
    clauses = filter(lambda x: 0 not in x, clauses)  # Clauses containing True are redundant
    clauses = map(lambda x: filter(lambda y: y is not None, x), clauses)  # Remove False in each clause
    clauses = filter(lambda x: x != [], clauses)  # Remove empty clauses
    return clauses


def build_totalizer(input_variables):
    # Build the totalizer formulae using the input variables given as the leaves of the tree
    # Returns the totalizer clauses, as well as the identifiers of the output variables at the top of the tree
    new_array = list(
        map(lambda x: [x], input_variables))  # transform a set like {1, 2, 3} into a list like [[1], [2], [3]]
    clause_list = []
    while len(new_array) != 1:
        n = []
        # Deal with odd arrays as a special case
        if len(new_array) % 2 == 1:
            retval, freshes = encode_sum(new_array[-1], new_array[-2])
            clause_list += retval
            new_array = new_array[:-1]
            new_array[-1] = freshes
        for v, w in zip(new_array[::2], new_array[1::2]):
            retval, freshes = encode_sum(v, w)
            clause_list += retval
            n.append(freshes)
        new_array = n
    finals = new_array[0]
    return clause_list, finals


def build_comparator(l, u, output_variables):
    # Take in an ordered list of the output_variables and produce clauses setting the appropriate constraint
    output_clauses = []

    for i in range(l):
        output_clauses.append([output_variables[i]])

    for i in range(u, len(output_variables)):
        output_clauses.append([-output_variables[i]])

    return output_clauses


# 1 + 0 = 10 (counting from left to right: r1, r2)
def encode_sum(v, w):
    formulae = []
    freshes = [fresh_variable() for i in range(len(v) + len(w))]
    nv = len(v) + 1
    nw = len(w) + 1
    nr = len(freshes) + 1
    freshes = [0] + freshes + [None]
    v = [0] + v + [None]
    w = [0] + w + [None]
    for a in range(nv):
        for b in range(nw):
            for s in range(nr):
                if a + b == s:
                    formulae.append([_neg(v[a]), _neg(w[b]), freshes[s]])
                    formulae.append([v[a + 1], w[b + 1], _neg(freshes[s + 1])])
    return formulae, freshes[1:-1]


def _neg(k):
    if k is None:
        return 0
    elif k == 0:
        return None
    else:
        return -k


def apply_sub(predicate, acs):
    new_args = []
    for v in predicate.args:
        if v in acs:
            new_args.append(acs[v])
        else:
            new_args.append(v)
    new_predicate = deepcopy(predicate)
    new_predicate.args = new_args
    return new_predicate


def get_free_variables(clause_list):
    vars = set()
    for clause in clause_list:
        for predicate in clause:
            for var in predicate.free_variables():
                vars.add(var)
    return vars


def fresh_variable():
    global varcount
    varcount += 1
    return varcount


def get_n_fresh(n):
    a = set()
    for i in range(n):
        a.add(fresh_variable())
    return a


def get_upper_lower_bound(constrains, weights, domainsize, arities):
    lower_bound = 1
    upper_bound = 1
    # multiple unbounded formula
    for p_aux, (c_l, c_u) in constrains.items():
        nog = domainsize ** arities[p_aux]
        l, u = constrains[p_aux]
        lower_bound *= min(
            weights[p_aux][0] ** l * weights[p_aux][1] ** (nog - l),
            weights[p_aux][0] ** u * weights[p_aux][1] ** (nog - u)
        )
        upper_bound *= max(
            weights[p_aux][0] ** l * weights[p_aux][1] ** (nog - l),
            weights[p_aux][0] ** u * weights[p_aux][1] ** (nog - u)
        )
    return lower_bound, upper_bound


def get_upper_lower_bound_imp(convex_hull, constrains, weights, aux2dim,
                              domainsize, arities):
    n_formulas = len(constrains)
    # [n_formulas]
    c = np.zeros(n_formulas)
    # [facets, n_formulas]
    A = np.hstack([
        convex_hull.equations[:, :-1],
        np.zeros([convex_hull.equations.shape[0],
                  n_formulas - convex_hull.equations.shape[1] + 1])
    ])
    b = -convex_hull.equations[:, -1]

    nog_log_sum = 0
    n_dims = len(aux2dim)

    for p_aux, (l, u) in constrains.items():
        nog = domainsize ** arities[p_aux]
        # ln (w_p^c * w_n^{nog-c}) = nog*ln(w_p) + ln(w_p/w_n) * c
        obj_w = math.log(weights[p_aux][0] / weights[p_aux][1])
        nog_log_sum += nog * math.log(weights[p_aux][1])
        dim = None
        if p_aux in aux2dim:
            dim = aux2dim[p_aux]
        else:
            dim = n_dims
            n_dims += 1
        c[dim] = obj_w

        A_p = np.zeros([1, A.shape[1]])
        A_p[0, dim] = 1
        A = np.vstack([A, -A_p, A_p])
        b = np.append(b, [-constrains[p_aux][0], constrains[p_aux][1]])
    try:
        res = linprog(c, A, b)
        if not res.success:
            logger.error(res)
        lower_bound = res.fun
        res = linprog(-c, A, b)
        if not res.success:
            logger.error(res)
        upper_bound = -res.fun
    except Exception as e:
        logger.error(e)
        raise RuntimeError('Encount error when calling linprog')

    lower_bound = math.exp(lower_bound + nog_log_sum)
    upper_bound = math.exp(upper_bound + nog_log_sum)
    # # multiple unbounded formula
    # for p_aux, (c_l, c_u) in constrains.items():
    #     if p_aux not in aux2dim:
    #         nog = domainsize ** arities[p_aux]
    #         l, u = constrains[p_aux]
    #         lower_bound *= min(
    #             weights[p_aux][0] ** l * weights[p_aux][1] ** (nog - l),
    #             weights[p_aux][0] ** u * weights[p_aux][1] ** (nog - u)
    #         )
    #         upper_bound *= max(
    #             weights[p_aux][0] ** l * weights[p_aux][1] ** (nog - l),
    #             weights[p_aux][0] ** u * weights[p_aux][1] ** (nog - u)
    #         )
    return lower_bound, upper_bound


def contains(convex_hull, point):
    return np.all(np.dot(convex_hull.equations, np.append(point, 1)) <= 0)


if __name__ == '__main__':
    main()
