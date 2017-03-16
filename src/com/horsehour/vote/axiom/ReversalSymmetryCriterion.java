package com.horsehour.vote.axiom;

import java.lang.reflect.Array;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.RankedPairs;
import com.horsehour.vote.rule.VotingRule;

/**
 * Reversal symmetry is a voting system criterion which requires that if
 * candidate A is the unique winner, and each voter's individual preferences are
 * inverted, then A must not be elected.
 * <p>
 * Methods that satisfy reversal symmetry include Borda count, the Kemeny-Young
 * method, and the Schulze method.
 * <p>
 * Methods that fail include Bucklin voting, instant-runoff voting and Condorcet
 * methods that fail the Condorcet loser criterion such as Minimax. For cardinal
 * voting systems which can be meaningfully reversed, approval voting and range
 * voting satisfy the criterion.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:30:24 PM, Jul 31, 2016
 *
 */

public class ReversalSymmetryCriterion extends VotingAxiom {
	BiPredicate<Integer, List<Integer>> violator = (winner, predicted) -> {
		if (predicted == null || predicted.contains(winner))
			return true;
		else
			return false;
	};

	@Override
	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {

			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null || winners.size() > 1)
				return;

			numTotal.addAndGet(1);

			Profile<Integer> invProfile = invertProfile(profile);
			List<Integer> invWinners = rule.getAllWinners(invProfile);
			if (!violator.test(winners.get(0), invWinners))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	@Override
	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null || winners.size() > 1)
				return false;

			Profile<Integer> invProfile = invertProfile(profile);
			List<Integer> invWinners = rule.getAllWinners(invProfile);
			if (violator.test(winners.get(0), invWinners))
				return true;
			else
				return false;
		});
	}

	/**
	 * @param profile
	 * @return inverted profile
	 */
	@SuppressWarnings("unchecked")
	<T> Profile<T> invertProfile(Profile<T> profile) {
		int numRanking = profile.votes.length;
		int numItem = profile.data[0].length;

		T[][] data = (T[][]) Array.newInstance(profile.data[0][0].getClass(), new int[] { numRanking, numItem });
		for (int i = 0; i < numRanking; i++)
			for (int j = 0; j < numItem; j++)
				data[i][j] = profile.data[i][numItem - j - 1];
		return new Profile<>(data, profile.votes);
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		VotingRule rule = null;
		// rule = new Condorcet();
		// rule = new Borda();
		rule = new RankedPairs();

		VotingAxiom axiom = null;
		axiom = new ReversalSymmetryCriterion();

		int numItem = 3, numVote = 7;
		System.out.println(axiom.getSatisfiability(numItem, numVote, rule));

		TickClock.stopTick();
	}
}