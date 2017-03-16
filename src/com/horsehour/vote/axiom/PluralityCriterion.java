package com.horsehour.vote.axiom;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Plurality;
import com.horsehour.vote.rule.VotingRule;

/**
 * Plurality criterion is a voting system criterion devised by Douglas R.
 * Woodall for ranked voting methods with incomplete ballots. It is stated as
 * follows: If the number of ballots ranking A as the first preference is
 * greater than the number of ballots on which another candidate B is given any
 * preference, then A's probability of winning must be no less than B's.
 * <p>
 * This criterion is trivially satisfied by rank ballot methods which require
 * voters to strictly rank all the candidates (and so do not allow truncation).
 * The Borda count is usually defined in this way.
 * <p>
 * Woodall has called the Plurality criterion "a rather weak property that
 * surely must hold in any real election", and noted that "every reasonable
 * electoral system seems to satisfy it." Most proposed methods do satisfy it,
 * including Plurality voting, IRV, Bucklin voting, and approval voting. Among
 * Condorcet methods which permit truncation, whether the Plurality criterion is
 * satisfied depends often on the measure of defeat strength. When winning votes
 * is used as the measure of defeat strength in methods such as the Schulze
 * method, Ranked Pairs, or Minimax, Plurality is satisfied. Plurality is failed
 * when margins is used. Minimax using pairwise opposition also fails Plurality.
 * <p>
 * When truncation is permitted under Borda count, Plurality is satisfied when
 * no points are scored to truncated candidates, and ranked candidates receive
 * no fewer votes than if the truncated candidates had been ranked. If truncated
 * candidates are instead scored the average number of points that would have
 * been awarded to those candidates had they been strictly ranked, or if Nauru's
 * modified Borda count is used, the Plurality criterion is failed.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:25:55 PM, Jul 31, 2016
 *
 */

public class PluralityCriterion extends VotingAxiom {
	Plurality plurality = new Plurality();

	BiPredicate<List<Integer>, List<Integer>> violator = (winners, predicted) -> {
		if (predicted == null)
			return true;
		// plurality winner(s) should be selected
		else if (winners.size() <= predicted.size())
			return winners.stream().anyMatch(c -> !predicted.contains(c));
		// selected candidates should also be plurality winner(s)
		else
			return predicted.stream().anyMatch(c -> !winners.contains(c));
	};

	@Override
	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			List<Integer> winners = plurality.getAllWinners(profile);
			if (winners == null)
				return;

			numTotal.addAndGet(1);

			List<Integer> predicted = rule.getAllWinners(profile);
			if (!violator.test(winners, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	@Override
	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			List<Integer> winners = plurality.getAllWinners(profile);
			if (winners == null)
				return false;

			List<Integer> predicted = rule.getAllWinners(profile);
			return violator.test(winners, predicted);
		});
	}
}
