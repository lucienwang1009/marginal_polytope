import argparse

from pyparsing import *
from itertools import product
from copy import deepcopy

i = 0


class Predicate:
    def __init__(self, name, args, negated):
        self.name = name
        self.args = args
        self.negated = negated

    def arity(self):
        return len(self.args)

    def free_variables(self):
        return list(filter(lambda x: isinstance(x, Variable), self.args))

    def ground_variables(self):
        return list(filter(lambda x: isinstance(x, Constant), self.args))

    def substitute(self, map):
        for key, value in map.items():
            self.args = map(lambda x: value if x == key else x, self.args)

    def __hash__(self):
        return hash((self.name, str(self.args)))

    def __eq__(self, other):
        return other and self.name == other.name and self.args == other.args

    def __repr__(self):
        out = self.name + "(" + ",".join(map(str, self.args)) + ")"
        if self.negated: out = "-" + out
        return out


class Variable:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return other and self.name == other.name


class Constant:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return other and self.name == other.name

    def __hash__(self):
        return hash(str(self.name))

    def __repr__(self):
        return self.name


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


def parseInput(input):
    variable = Word(alphas.upper(), alphanums)
    predicate_name = Word(alphanums + '_' + '-')
    ground_atom = Word(nums)
    decimal_number = Word(nums + '.' + '-')

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

    args = parser.parse_args()
    with open(args.file, 'r') as file:
        data = file.read()
    domainsize = args.domain
    d = parseInput(data)

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
            c.append(p)
        clauses.append(c)

    # At this point, clauses contains a list of lists of predicates
    sig = {}
    weights = {}
    for decl in d[0]:
        sig[decl[0]] = int(decl[1])
        weights[decl[0]] = (float(decl[2]), float(decl[3]))
    free_variables = list(get_free_variables(clauses))

    subs = product(map(lambda l: Constant(str(l)), range(domainsize)), repeat=len(free_variables))
    ground_clauses = []
    for sub in subs:
        acs = {}
        for q, var in enumerate(free_variables):
            acs[Variable(var.name)] = sub[q]
        ground_clauses += [list(map(lambda pr: apply_sub(pr, acs), clause)) for clause in clauses]

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

    m = {}
    for k, v in index.items():
        if k.name not in m:
            m[k.name] = [v]
        else:
            m[k.name] = m[k.name] + [v]

    my_dic = {k: domainsize ** v for k, v in sig.items()}
    pairs = []
    for k, v in my_dic.items():
        o = [(k, p) for p in range(v + 1)]
        pairs.append(o)

    add_weights = False  # Whether to add weights or not
    pysddmode = False  # Whether to print the weights in PySDD format or not

    print(output_to_dimacs(out))
    if add_weights:
        if pysddmode:
            outstr = "c weights"
            for i, j in index.items():
                weight = weights[i.name]
                outstr += " " + str(weight[0]) + " " + str(weight[1])
            print(outstr)
        else:
            for i, j in index.items():
                weight = weights[i.name]
                if weight[0] != 1:
                    print("w", j, weight[0])
                if weight[1] != 1:
                    print("w", -j, weight[1])


def output_to_dimacs(clauses):
    p = "p cnf " + str(i) + " " + str(len(clauses)) + "\n"
    p += '\n'.join(map(str, clauses))
    return p


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


if __name__ == '__main__':
    main()
