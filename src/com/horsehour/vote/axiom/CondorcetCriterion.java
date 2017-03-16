package com.horsehour.vote.axiom;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Condorcet;
import com.horsehour.vote.rule.VotingRule;

/**
 * A Condorcet winner (cw) is a candidate in a voting preference profile, who
 * could beat each of other candidates in head-to-head comparisons based on
 * majority rule. If a voting rule satisfies Condorcet consistency criterion, we
 * say that it must select cw from a profile if (s)he exists.
 * <p>
 * The above illustration is the definition of the Condorcet winner. If a
 * candidate beats or ties with every other candidate in a pairwise matchup,
 * it's called weak Condorcet winner. There can be more than one weak Condorcet
 * winner.
 * <p>
 * No positional scoring rule is Condorcet consistent.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 10:44:27 AM, Jun 17, 2016
 *
 */

public class CondorcetCriterion extends VotingAxiom {
	Condorcet condorcet = new Condorcet();

	/**
	 * The Condorcet winner is neither selected nor the only one selected
	 */
	BiPredicate<Integer, List<Integer>> violator = (winner, predicted) -> {
		if (predicted == null || predicted.size() > 1)
			return true;
		if (predicted.get(0) == winner)
			return false;
		else
			return true;
	};

	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotalCP = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			Integer winner = condorcet.getWinner(profile);
			if (winner == null)
				return;
			// long stat = profile.getStat();
			numTotalCP.addAndGet(1);
			List<Integer> predicted = rule.getAllWinners(profile);
			if (!violator.test(winner, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotalCP.doubleValue();
	}

	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			Integer winner = condorcet.getWinner(profile);
			if (winner == null)
				return false;
			List<Integer> predicted = rule.getAllWinners(profile);
			return violator.test(winner, predicted);
		});
	}
}