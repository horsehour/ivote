package com.horsehour.vote.axiom;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Majority;
import com.horsehour.vote.rule.VotingRule;

/**
 * The majority loser criterion is a criterion to evaluate single-winner voting
 * systems. The criterion states that if a majority of voters prefers every
 * other candidate over a given candidate, then that candidate must not win.
 * <p>
 * Either of the Condorcet loser criterion or the mutual majority criterion
 * imply the majority loser criterion. However, the Condorcet criterion does not
 * imply the majority loser criterion. Neither does the majority criterion imply
 * the majority loser criterion.
 * <p>
 * Methods that comply with this criterion include Schulze, Ranked Pairs,
 * Kemeny-Young, Nanson, Baldwin, Coombs, Borda, Bucklin, instant-runoff voting,
 * contingent voting, and anti-plurality voting.
 * <p>
 * Methods that do not comply with this criterion include plurality, MiniMax,
 * Sri Lankan contingent voting, supplementary voting, approval voting, and
 * range voting.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:22:50 PM, Jul 31, 2016
 *
 */

public class MajorityLoserCriterion extends VotingAxiom {
	Majority majority = new Majority();

	BiPredicate<Integer, List<Integer>> violator = (loser, predicted) -> {
		if (predicted == null || predicted.contains(loser))
			return true;
		else
			return false;
	};

	@Override
	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			// majority loser
			Integer loser = majority.getLoser(profile);
			if (loser == null)
				return;

			numTotal.addAndGet(1);

			List<Integer> predicted = rule.getAllWinners(profile);
			if (!violator.test(loser, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	@Override
	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			// majority loser
			Integer loser = majority.getLoser(profile);
			if (loser == null)
				return false;

			List<Integer> predicted = rule.getAllWinners(profile);
			return violator.test(loser, predicted);
		});
	}
}
