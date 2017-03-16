package com.horsehour.vote.axiom;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.VotingRule;

/**
 * The Homogeneity criterion is a voting system property formulated by Douglas
 * Woodall. The property is satisfied if, in any election, the result depends
 * only on the proportion of ballots of each possible type. Specifically, if
 * every ballot is replicated the same number of times, then the result should
 * not change. Woodall considers Homogeneity to be one of several voting method
 * properties "sufficiently basic to deserve to be called axioms."
 * <p>
 * Any voting method that counts voter preferences proportionally satisfies
 * Homogeneity, including voting methods such as Plurality voting, Two-round
 * system, Single transferable vote, Instant Runoff Voting, Contingent vote,
 * Coombs' method, Approval voting, Anti-plurality voting, Borda count, Range
 * voting, Bucklin voting, Majority Judgment, Condorcet methods and others.
 * <p>
 * A voting method that determines a winner by eliminating candidates not having
 * a fixed number of votes, rather than a proportional or a percentage of votes,
 * may not satisfy the Homogeneity criterion.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 1:38:32 AM, Aug 4, 2016
 *
 */

public class HomogeneityCriterion extends VotingAxiom {
	/**
	 * After all ballots are increased by the same number of votes, the winners
	 * change
	 */
	BiPredicate<List<Integer>, List<Integer>> violator = (winners, predicted) -> {
		if (predicted == null)
			return true;
		else if (predicted.size() <= winners.size())
			return predicted.stream().anyMatch(c -> !winners.contains(c));
		else
			return winners.stream().anyMatch(c -> !predicted.contains(c));
	};

	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			int numVote = profile.numVoteTotal;
			numTotal.addAndGet(numVote);

			List<Integer> truth = rule.getAllWinners(profile);
			List<Integer> predict = null;
			for (int copy = 1; copy <= numVote; copy++) {
				predict = rule.getAllWinners(new Profile<>(profile.data, MathLib.Matrix.multiply(profile.votes, copy)));
				if (!violator.test(truth, predict))
					numMatch.addAndGet(1);
			}
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	@Override
	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			int numVote = profile.numVoteTotal;

			List<Integer> truth = rule.getAllWinners(profile);
			List<Integer> predict = null;
			for (int copy = 1; copy <= numVote; copy++) {
				predict = rule.getAllWinners(new Profile<>(profile.data, MathLib.Matrix.multiply(profile.votes, copy)));
				if (violator.test(truth, predict))
					return true;
			}
			return false;
		});
	}
}