import argparse
import random
import logzero

from pracmln import MLN, Database
from contexttimer import Timer
from logzero import logger

from map_inference import map_inference
from main import construct_polytope

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, help='input mln file', required=True)
parser.add_argument('--evidence_percentage', '-p', type=float,
                    help='evidence percentage', required=True)
parser.add_argument('--log', '-l', type=str, help='log file')
args = parser.parse_args()

logzero.logfile(args.log, mode='w')

positive_evidence_ratio = args.evidence_percentage / 2
negative_evidence_ratio = args.evidence_percentage / 2
n_repeats = 10
mln = MLN.load(args.input, grammar='StandardGrammar')
rmp = construct_polytope(mln)
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
