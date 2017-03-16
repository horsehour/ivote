package com.horsehour.vote.axiom;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Borda;
import com.horsehour.vote.rule.Majority;
import com.horsehour.vote.rule.VotingRule;

/**
 * The majority criterion is a single-winner voting system criterion, used to
 * compare such systems. The criterion states that "if one candidate is
 * preferred by a majority (more than 50%) of voters, then that candidate must
 * win".
 * <p>
 * Some methods that comply with this criterion include any Condorcet method,
 * instant-runoff voting, and Bucklin voting.
 * <p>
 * Some methods which give weight to preference strength fail the majority
 * criterion, while others pass it. Thus the Borda count and range voting fail
 * the majority criterion, while the Majority judgment passes it. The
 * application of the majority criterion to methods which cannot provide a full
 * ranking, such as approval voting, is disputed.
 * <p>
 * These methods that fail the majority criterion may offer a strategic
 * incentive to voters to bullet vote, i.e., vote for one candidate only, not
 * providing any information about their possible support for other candidates,
 * since, with such methods, these additional votes may aid their
 * less-preferred.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:22:50 PM, Jul 31, 2016
 *
 */

public class MajorityCriterion extends VotingAxiom {
	Majority majority = new Majority();

	BiPredicate<Integer, List<Integer>> violator = (winner, predicted) -> {
		if (predicted == null || !predicted.contains(winner))
			return true;
		else
			return false;
	};

	@Override
	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			// majority winner
			Integer winner = majority.getWinner(profile);
			if (winner == null)
				return;

			numTotal.addAndGet(1);

			List<Integer> predicted = rule.getAllWinners(profile);
			if (!violator.test(winner, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	@Override
	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			// majority winner
			Integer winner = majority.getWinner(profile);
			if (winner == null)
				return false;

			List<Integer> predicted = rule.getAllWinners(profile);
			return violator.test(winner, predicted);
		});
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		VotingRule rule = null;
		rule = new Borda();
		// rule = new Plurality();
		// rule = new InstantRunoff();
		// rule = new Condorcet();

		VotingAxiom axiom = new MajorityCriterion();

		int numItem = 3, numVote = 11;
		System.out.println(axiom.getSatisfiability(numItem, numVote, rule));

		TickClock.stopTick();
	}
}