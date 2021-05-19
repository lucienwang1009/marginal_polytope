import argparse
import logzero
import os
import numpy as np

from pracmln import MLN, Database
from mip import Model, MAXIMIZE, BINARY, OptimizationStatus, CBC
from logzero import logger
from contexttimer import Timer

from main import construct_surrogate_polytope
from utils import get_hyperplane


def parse_args():
    parser = argparse.ArgumentParser(
        description='MAP inference'
    )
    parser.add_argument('--mln', '-m', type=str, required=True,
                        help='mln file')
    parser.add_argument('--database', '-d', type=str, required=True,
                        help='database file containing evidences')
    parser.add_argument('--out_dir', '-o', type=str, default='./checkpoint',
                        help='output directory for results and logs')
    parser.add_argument('--use_rmp', action='store_true')
    parser.add_argument('--rmp_max_vertices', type=int, default=None)
    args = parser.parse_args()
    return args


def constrain_by_grounding_formula(model, ground_formula):
    weight = float(ground_formula.weight)
    lhs = []
    var = None
    n_literals = 0
    for literal in ground_formula.nnf().literals():
        if literal.negated:
            lhs.append(1 - model.var_by_name(atom2str(literal.gndatoms()[0])))
        else:
            lhs.append(model.var_by_name(atom2str(literal.gndatoms()[0])))
        n_literals += 1
    if ground_formula.ishard:
        model += sum(lhs) >= 1
    else:
        if weight > 0:
            var = model.add_var(formula2str(ground_formula), var_type=BINARY)
            model += sum(lhs) >= var
        elif weight < 0:
            var = model.add_var(formula2str(ground_formula), var_type=BINARY)
            model += sum(lhs) <= n_literals * var
        else:
            logger.info('weight is 0, skip the ground formula')
    return var


def atom2str(atom):
    return 'A_' + str(atom)


def formula2str(formula):
    return 'F_' + str(formula)


def halfspace_with_integral_norm(rmp):
    halfspace = []
    for idx, s in enumerate(rmp.simplices):
        vertices = [np.array(rmp.points[i], dtype=np.int32) for i in s]
        norm, intercept = get_hyperplane(vertices)
        if rmp.equations[idx][-1] * intercept >= 0:
            norm = [-i for i in norm]
            intercept = -intercept
        halfspace.append([norm, intercept])
    return halfspace


def constrain_by_rmp(model, mrf, rmp):
    formulae = mrf.mln.formulas
    halfspace = halfspace_with_integral_norm(rmp)
    ground_formulae_satisfication = []
    for formula in formulae:
        satisfication = []
        for ground_formula in formula.itergroundings(mrf):
            satisfication.append(
                model.var_by_name(formula2str(ground_formula))
            )
        ground_formulae_satisfication.append(sum(satisfication))
    for norm, intercept in halfspace:
        constr = sum(n_f * a_i for n_f, a_i in
                     zip(ground_formulae_satisfication, norm)) <= intercept
        model += constr


def map_inference(mln, database, rmp=None):
    m = Model(sense=MAXIMIZE, solver_name=CBC)

    mrf = mln.ground(database)
    # add new vars associated to ground atoms
    for ground_atom in mrf.gndatoms:
        m.add_var(atom2str(ground_atom), var_type=BINARY)
    # constrain by evidences
    for e, is_true in mrf.db.evidence.items():
        if is_true:
            m += m.var_by_name(atom2str(e)) >= 1
        else:
            m += m.var_by_name(atom2str(e)) <= 0

    # constrain by ground formulae
    weights = []
    formula_vars = []
    for ground_formula in mrf.itergroundings():
        var = constrain_by_grounding_formula(m, ground_formula)
        if var is not None:
            formula_vars.append(var)
            weights.append(ground_formula.weight)

    # constrain by rmp
    if rmp is not None:
        constrain_by_rmp(m, mrf, rmp)

    # objective
    m.objective = sum(w * v for w, v in zip(weights, formula_vars))

    with Timer() as t:
        status = m.optimize()
    logger.debug('elapsed time for ILP: %s', t.elapsed)
    if status == OptimizationStatus.OPTIMAL:
        logger.info('optimal solution found: %s', m.objective_value)
    elif status == OptimizationStatus.FEASIBLE:
        logger.info('sol.cost %s found, best possible: %s',
                    m.objective_value, m.objective_bound)
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        logger.info('no feasible solution found, lower bound is: %s',
                    m.objective_bound)
    # if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    #     logger.info('solution:')
    #     for v in m.vars:
    #         logger.info('%s: %s', v.name, v.x)
    return m.objective_value


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logzero.logfile(os.path.join(args.out_dir, 'log.txt'))
    logzero.loglevel(logzero.logging.DEBUG)

    mln = MLN.load(args.mln, grammar='StandardGrammar')
    rmp = None
    if args.use_rmp:
        rmp = construct_surrogate_polytope(mln, max_vertices=args.rmp_max_vertices)

    db = Database(mln, dbfile=args.database)
    map_inference(mln, db, rmp)
