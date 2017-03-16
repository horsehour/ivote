package com.horsehour.vote.rule;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ProfileList;

import ilog.concert.IloConstraint;
import ilog.concert.IloException;
import ilog.concert.IloIntExpr;
import ilog.concert.IloIntVar;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.concert.IloObjectiveSense;

public class MinFirstMistakesRule<T> extends OptimizedPositionalRule<T> {

	@Override
	public void optimize(ProfileList<T> preferences, double normalization) throws IloException {
		super.optimize(preferences, normalization);
		// Want the first total to be higher than all the others

		IloIntExpr[] flipArr = new IloIntExpr[preferences.size()];
		IloNumExpr[] minMarginsNorm = new IloNumExpr[preferences.size()];

		int p = 0;
		for (Profile<T> profile : preferences) {
			IloLinearNumExpr[] totals = profileTotals.get(p);

			T[] choices = profile.getSortedItems();

			IloIntVar[] flips = new IloIntVar[numCandidates - 1];

			double currentMaxPossibleMargin = normalization * profile.getNumVote();
			IloNumVar currentMinMargin = cp.numVar(0, Double.POSITIVE_INFINITY);

			for (int i = 1; i < numCandidates; i++) {
				IloConstraint mistake = cp.le(totals[0], totals[i]);

				IloIntVar flip = cp.boolVar(choices[0] + ">" + choices[i]);
				IloConstraint flipGe1 = cp.ge(flip, 1);
				cp.add(cp.ifThen(mistake, flipGe1));

				IloNumExpr margin = cp.diff(totals[0], totals[i]);
				IloNumExpr condConstr = cp.sum(cp.prod(flip, currentMaxPossibleMargin), margin);
				cp.addLe(currentMinMargin, condConstr);

				flips[i - 1] = flip;
			}

			flipArr[p] = cp.sum(flips);

			minMarginsNorm[p] = cp.prod(currentMinMargin, -1.0 / currentMaxPossibleMargin / preferences.size());

			p++;
		}

		cp.addObjective(IloObjectiveSense.Minimize, cp.sum(cp.sum(flipArr), cp.sum(minMarginsNorm)));

		cp.solve();

		scores = cp.getValues(posScores);
	}

}
