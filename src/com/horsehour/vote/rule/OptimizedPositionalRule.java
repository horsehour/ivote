package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ProfileList;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

/**
 * GUROBI: most efficient optimization library
 * <p>
 * CPLEX: very promising optimization library owned by IBM
 * http://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.0/ilog.odms.cplex.help
 * /CPLEX/GettingStarted/topics/set_up/Eclipse.html
 * 
 * @author Andrew Mao
 */
public abstract class OptimizedPositionalRule<T> extends PositionalVotingRule {

	protected double[] scores;

	protected ProfileList<?> preferences;
	protected double normalization;

	IloCplex cp;

	int numCandidates;
	IloNumVar[] posScores;

	List<Map<T, int[]>> profileCounts;
	List<IloLinearNumExpr[]> profileTotals;

	protected OptimizedPositionalRule() {
		try {
			cp = new IloCplex();
		} catch (IloException e) {
			throw new RuntimeException(e);
		}

		// Turn comments on or off
		cp.setOut(null);
	}

	protected int getNumPairs() {
		return numCandidates * (numCandidates - 1) / 2;
	}

	@Override
	protected double[] getPositionalScores(int length) {
		return scores;
	}

	public double[][] getTotalScores() throws IloException {
		double[][] allScores = new double[profileTotals.size()][];

		int i = 0;
		for (IloLinearNumExpr[] totals : profileTotals) {
			allScores[i] = new double[totals.length];
			for (int j = 0; j < totals.length; j++)
				allScores[i][j] = cp.getValue(totals[j]);
			i++;
		}

		return allScores;
	}

	public void optimize(ProfileList<T> preferences, double normalization) throws IloException {
		this.preferences = preferences;
		this.normalization = normalization;

		profileCounts = new ArrayList<>(preferences.size());
		profileTotals = new ArrayList<>(preferences.size());

		numCandidates = preferences.getCandidates().length;

		cp.clearModel();

		posScores = cp.numVarArray(numCandidates, 0, Double.POSITIVE_INFINITY);

		// Add normalization constraint
		cp.addEq(cp.sum(posScores), normalization);

		// Add monotonicity constraint
		for (int i = 0; i < posScores.length - 1; i++) {
			cp.addGe(posScores[i], posScores[i + 1]);
		}

		for (Profile<T> profile : preferences) {
			// Count the number of times each preference appears in each
			// position

			// This map induces the ordering on the T. Its comparator is
			// important!
			Map<T, int[]> counts = profile.getPositionCounts();

			IloLinearNumExpr[] totals = new IloLinearNumExpr[numCandidates];

			int i = 0;
			for (Map.Entry<T, int[]> e : counts.entrySet()) {
				int[] count = e.getValue();
				totals[i++] = cp.scalProd(count, posScores);
			}

			// Save each of the variables we created
			profileCounts.add(counts);
			profileTotals.add(totals);
		}
	}
}
