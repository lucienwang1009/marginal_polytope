import argparse
import random
import logzero

from pracmln import MLN, Database
from contexttimer import Timer
from logzero import logger

from map_inference import map_inference
from main import construct_surrogate_polytope

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, help='input mln file', required=True)
parser.add_argument('--domain_size', '-d', type=int, help='domain size', required=True)
parser.add_argument('--log', '-l', type=str, help='log file')
args = parser.parse_args()

logzero.logfile(args.log, mode='w')

positive_evidence_ratio = 0.05
negative_evidence_ratio = 0.05
n_repeats = 3
mln = MLN.load(args.input, grammar='StandardGrammar')
domain_name = list(mln.domains)[0]
mln.domains[domain_name] = [str(i) for i in range(args.domain_size)]
rmp = construct_surrogate_polytope(mln, max_vertices=10)
db = Database(mln)
mrf = mln.ground(db)

total_ground_atoms = mrf.gndatoms
n_evidences = len(total_ground_atoms)
n_positive_samples = int(n_evidences * positive_evidence_ratio)
n_negative_samples = int(n_evidences * negative_evidence_ratio)
for i in range(n_repeats):
    sampled_evidences = random.sample(
        total_ground_atoms,
        n_positive_samples + n_negative_samples
    )
    random.shuffle(sampled_evidences)
    positive_evidences = sampled_evidences[:n_positive_samples]
    negative_evidences = sampled_evidences[n_positive_samples:]
    evidences = dict([(str(g), True) for g in positive_evidences] +
                     [(str(g), False) for g in negative_evidences])
    sampled_db = Database(mln, evidence=evidences)
    logger.info('MAP with RMP')
    optimal_with_rmp = map_inference(mln, sampled_db, rmp)
    logger.info('MAP without RMP')
    optimal_without_rmp = map_inference(mln, sampled_db)
    if abs(optimal_with_rmp - optimal_without_rmp) > 1e-6:
        raise RuntimeError('Inconsistent optimal objective!')
