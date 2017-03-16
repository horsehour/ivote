package com.horsehour.vote.axiom;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.Condorcet;
import com.horsehour.vote.rule.VotingRule;

/**
 * The Schwartz set in voting system is the union of all Schwartz set
 * components, and a Schwartz set component is any non-empty set S of candidates
 * such that: (1) Every candidate inside the set S is pairwise unbeaten by every
 * candidate outside S; and (2) No non-empty proper subset of S fulfills the
 * first property.
 * <p>
 * A set of candidates that meets the first requirement is also known as an
 * undominated set. The Schwartz set provides one standard of optimal choice for
 * an election outcome. Voting systems that always elect a candidate from the
 * Schwartz set pass the Schwartz criterion. The Schwartz set is named after
 * political scientist Thomas Schwartz.
 * <p>
 * The Schwartz set is closely related to and is always a subset of the Smith
 * set. The Smith set is larger if and only if a candidate in the Schwartz set
 * has a pairwise tie with a candidate that is not in the Schwartz set. The
 * Schwartz set can be calculated with the Floyd–Warshall algorithm in time
 * Θ(n^3) or with a version of Kosaraju's algorithm in time Θ(n^2).
 * <p>
 * The Schulze method always chooses a winner from the Schwartz set.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:21:14 PM, Aug 8, 2016
 *
 */

public class SchwarzCriterion extends SmithCriterion {

	@Override
	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		AtomicLong numTotalCP = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			List<Integer> schwarzSet = getSchwarzSet(profile);
			if (schwarzSet == null)
				return;

			// long stat = profile.getStat();
			numTotalCP.addAndGet(1);

			List<Integer> predicted = rule.getAllWinners(profile);
			if (!violator.test(schwarzSet, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotalCP.doubleValue();
	}

	@Override
	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		return profiles.anyMatch(profile -> {
			List<Integer> schwarzSet = getSchwarzSet(profile);
			if (schwarzSet == null)
				return false;
			List<Integer> predicted = rule.getAllWinners(profile);
			return violator.test(schwarzSet, predicted);
		});
	}

	int numItem;

	<T> List<T> getSchwarzSet(Profile<T> profile) {
		T[] sortedItems = profile.getSortedItems();
		numItem = sortedItems.length;
		int[][] ppm = Condorcet.getPairwisePreferenceMatrix(profile, sortedItems);
		boolean[][] relation = new boolean[numItem][numItem];
		for (int i = 0; i < numItem; i++)
			for (int j = 0; j < numItem; j++)
				relation[i][j] = ppm[i][j] >= ppm[j][i];
		return getMaximalCandidates(sortedItems, relation);
	}

	public String toString() {
		return getClass().getSimpleName();
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		VotingAxiom axiom = new SchwarzCriterion();
		VotingRule rule = null;
		// rule = new RankedPairs();
		// rule = new Nanson();
		// rule = new Schulze();
		// rule = new Borda();
		rule = new Condorcet();

		System.out.println(axiom.getSatisfiability(3, 5, rule));

		TickClock.stopTick();
	}

}
