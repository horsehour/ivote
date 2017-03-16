package com.horsehour.vote.rule;

/**
 * Plurality (a.k.a First Past the Post or a Winner-Take-All)
 * 
 * <p>
 * <b>Criteria summary</b>
 * <li>Satisfied: majority, monotonicity, participation, consistency, Pareto,
 * later preferences</li>
 * <li>Violated: mutual majority, Condorcet, Condorcet loser, Smith,
 * independence of clones</li>
 * <li>Strategic vulnerability: Very strong and very damaging
 * compromising-reversal incentive.</li>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 */
public class Plurality extends PositionalVotingRule {

	@Override
	protected double[] getPositionalScores(int length) {
		double[] scores = new double[length];
		scores[0] = 1;
		return scores;
	}
}
