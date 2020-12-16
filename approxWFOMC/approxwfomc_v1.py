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
from approxWFOMC.approxwfomc import get_upper_lower_bound,\
    get_upper_lower_bound_imp
from main import main as get_irmp
from utils import plot_convex_hull

i = 0

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

def get_approxmc_model_count(input_str):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
        tf.write(input_str)
        tf.flush()
        out = subprocess.run(['./approxmc', tf.name], check=False,
                             stdout=subprocess.PIPE,
                             universal_newlines=True).stdout
        cell, has = re.findall(r'is: (\d+) x 2\^(\d+)', out.split('\n')[-2])[0]
        return int(cell) * math.pow(2, int(has))


get_model_count = lambda x: get_approxmc_model_count(x)
heuristic = lambda lb, ub, mc: -(ub - lb)
# heuristic = lambda ub, lb, mc: -mc

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
    parser.add_argument('--mln', action='store_true')
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

    args = parser.parse_args()
    with open(args.file, 'r') as file:
        data = file.read()
    domainsize = args.domain
    mlnmode = args.mln

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
    sampling_set = []
    non_aux_predicates = []
    aux_predicates = []

    if mlnmode:
        logger.info("MLN mode on")
        for pred in weights.keys():
            if not pred.startswith("aux"):
                non_aux_predicates.append(pred)
                sampling_set += m[pred]
            else:
                aux_predicates.append(pred)
    else:
        logger.info("MLN mode off")
        aux_predicates = weights.keys()

    start = time.time()
    mc = get_model_count(output_to_dimacs(out, sampling_set))
    number_of_mc_calls = 1

    priority_queue = []
    tmin = 1
    tmax = 1
    bounds = {}
    global i
    tolerance = 1.5

    for pred in aux_predicates:
        nog = domainsize ** arities[pred]

        # tolerance = 1
        p = weights[pred][0]
        n = weights[pred][1]

        minWeight = min(p ** nog, n ** nog)
        maxWeight = max(p ** nog, n ** nog)
        tmin *= minWeight
        tmax *= maxWeight
        bounds[pred] = (0, nog)

    newLb = mc * tmin
    newUb = mc * tmax
    heapq.heappush(priority_queue, (heuristic(newLb, newUb, mc), id(bounds), mc, bounds, newLb, newUb))
    # end init

    while True:
        if not (newUb / newLb > tolerance):
            logger.info("Converged!")
            break
        if not priority_queue:
            logger.info("Queue empty, terminating!")
            break
        logger.info("Current bounds on WMC: [" + str(newLb) + ", " + str(newUb) + "]")
        logger.info("Number of model counter calls so far: %s", number_of_mc_calls)
        logger.info("Current queue: %s", priority_queue)
        logger.info("Popping the first (left-most) item off the queue.")
        (_, parent_interval_mc, _, bounds, lb, ub) = heapq.heappop(priority_queue)
        # if themin == themax:
        #     continue

        newbounds = {}
        allsame = True
        # for pred in weights.keys():
        for pred in aux_predicates:
            themin, themax = bounds[pred]
            if(themin == themax):
                newbounds[pred] = [(themin, themax)]
            else:
                newbounds[pred] = [(themin,
                                    floor((themin + themax) / 2)),
                                   (floor((themin + themax) / 2) + 1,
                                    themax)]
                allsame = False
        if allsame:
            # No further splits possible, so we just ignore this item
            continue

        # We're tightening the bounds, so remove the coarser bounds to start
        newLb = newLb - lb
        newUb = newUb - ub

        logger.info(newbounds)
        for bounds in dict_product(newbounds):
            tmin = 1
            tmax = 1
            logger.info("Setting constraints: %s", bounds)
            # for pred in weights.keys():
            for pred in aux_predicates:
                p = weights[pred][0]
                n = weights[pred][1]
                themin, themax = bounds[pred]
                nog = domainsize ** arities[pred]
                minWeight = min(p ** themin * n ** (nog - themin), p ** themax * n ** (nog - themax))
                maxWeight = max(p ** themin * n ** (nog - themin), p ** themax * n ** (nog - themax))
                tmin *= minWeight
                tmax *= maxWeight

            tout = deepcopy(out)
            oldi = i
            # for pred in weights.keys():
            for pred in aux_predicates:
                amin, amax = bounds[pred]
                for x in encode_cardinality_constraint(amin, amax, m[pred]):
                    tout.add(DimacsClause(*list(x)))
            mcA = get_model_count(output_to_dimacs(tout, sampling_set))
            number_of_mc_calls += 1
            i = oldi
            logger.info("Model count for constraints above is: %s", mcA)
            if mcA == 0:
                continue
            lbA = mcA * tmin
            ubA = mcA * tmax

            newLb += lbA
            newUb += ubA
            heapq.heappush(priority_queue, (heuristic(lbA, ubA, mcA), id(bounds), mcA, bounds, lbA, ubA))

    logger.info("Total number of model counter calls: %s", number_of_mc_calls)
    logger.info("Best WMC bounds: [" + str(newLb) + ", " + str(newUb) + "]")
    end = time.time()
    logger.info("Runtime: %s", end - start)

# Borrowed from Stack Overflow: https://stackoverflow.com/a/40623158
def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def output_to_dimacs(clauses, sampling_set=None):
    p = "p cnf " + str(i) + " " + str(len(clauses)) + "\n"
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
    global i
    i += 1
    return i


def get_n_fresh(n):
    a = set()
    for i in range(n):
        a.add(fresh_variable())
    return a


if __name__ == '__main__':
    main()
