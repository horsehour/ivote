package com.horsehour.vote.axiom;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Condorcet;
import com.horsehour.vote.rule.VotingRule;

/**
 *
 * In single-winner voting system theory, the Condorcet loser criterion is a
 * measure for differentiating voting systems. It implies the majority loser
 * criterion.
 * <p>
 * A voting system complying with the Condorcet loser criterion will never allow
 * a Condorcet loser to win. A Condorcet loser is a candidate who can be
 * defeated in a head-to-head competition against each other candidate. (Not all
 * elections will have a Condorcet loser since it is possible for three or more
 * candidates to be mutually defeatable in different head-to-head competitions.)
 * <p>
 * A slightly weaker (easier to pass) version is the majority Condorcet loser
 * criterion, which requires that a candidate who can be defeated by a majority
 * in a head-to-head competition against each other candidate, lose. It is
 * possible for a system, such as Majority Judgment, which allows voters not to
 * state a preference between two candidates, to pass the MCLC but not the CLC.
 * <p>
 * Compliant methods include: two-round system, instant-runoff voting,
 * contingent vote, borda count, Schulze method, and ranked pairs. Noncompliant
 * methods include: plurality voting, supplementary voting, Sri Lankan
 * contingent voting, approval voting, range voting, Bucklin voting and minimax
 * Condorcet.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:37:02 PM, Jul 31, 2016
 *
 */

public class CondorcetLoserCriterion extends VotingAxiom {
	Condorcet condorcet = new Condorcet();

	/**
	 * The Condorcet loser is selected
	 */
	BiPredicate<Integer, List<Integer>> violator = (loser, predicted) -> {
		if (predicted == null || predicted.contains(loser))
			return true;
		else
			return false;
	};

	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotalCP = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			Integer loser = condorcet.getLoser(profile);
			if (loser == null)
				return;

			// long stat = profile.getStat();
			numTotalCP.addAndGet(1);

			List<Integer> predicted = rule.getAllWinners(profile);
			if (!violator.test(loser, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotalCP.doubleValue();
	}

	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			Integer loser = condorcet.getLoser(profile);
			if (loser == null)
				return false;

			List<Integer> predicted = rule.getAllWinners(profile);
			return violator.test(loser, predicted);
		});
	}
}
