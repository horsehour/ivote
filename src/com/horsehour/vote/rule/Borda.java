package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;

/**
 * The Borda count is a single-winner election method in which voters rank
 * options or candidates in order of preference. The Borda count determines the
 * outcome of a debate or the winner of an election by giving each candidate,
 * for each ballot, a number of points corresponding to the number of candidates
 * ranked lower. Once all votes have been counted the option or candidate with
 * the most points is the winner. Because it sometimes elects broadly acceptable
 * options or candidates, rather than those preferred by a majority, the Borda
 * count is often described as a consensus-based voting system rather than a
 * majoritarian one.
 * <p>
 * The Borda count was developed independently several times, but is named for
 * the 18th-century French mathematician and political scientist Jean-Charles de
 * Borda, who devised the system in 1770.
 * <p>
 * The number of points given to candidates for each ranking is determined by
 * the number of candidates standing in the election. Thus, under the simplest
 * form of the Borda count, if there are five candidates in an election then a
 * candidate will receive five points each time they are ranked first, four for
 * being ranked second, and so on, with a candidate receiving 1 point for being
 * ranked last (or left unranked). In other words, where there are n candidates
 * a candidate will receive n points for a first preference, n − 1 points for a
 * second preference, n − 2 for a third, and so on. Alternatively, votes can be
 * counted by giving each candidate a number of points equal to the number of
 * candidates ranked lower than them, so that a candidate receives n − 1 points
 * for a first preference, n − 2 for a second, and so on, with zero points for
 * being ranked last (or left unranked). In other words, a candidate ranked in
 * ith place receives n−i points. The candidate who gets the most points wins.
 * <p>
 * The Borda count can be combined with an instant-runoff procedure to create
 * hybrid election methods that are called Nanson method and Baldwin method.
 * <li>Nanson's method eliminates those choices from a Borda count tally that
 * are at or below the average Borda count score, then the ballots are retallied
 * as if the remaining candidates were exclusively on the ballot. This process
 * is repeated if necessary until a single winner remains.</li>
 * <li>Baldwin method tallies candidates' Borda scores in a series of rounds. In
 * each round, the candidate with the fewest points is eliminated, and the
 * points are re-tallied as if that candidate were not on the ballot.</li>
 */
public class Borda extends VotingRule {
	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] scores = getScores(profile, items);
		int[] rank = MathLib.getRank(scores, false);
		/**
		 * social preference
		 */
		List<T> preference = new ArrayList<>();
		for (int i : rank)
			preference.add(items[i]);
		return preference;
	}

	public <T> List<T> getAllWinners(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] scores = getScores(profile, items);
		int[] argmax = MathLib.argmax(scores);

		/**
		 * select all candidates with the largest Borda score
		 */
		List<T> winners = new ArrayList<>();
		for (int i : argmax)
			winners.add(items[i]);
		return winners;

		/**
		 * select one single winner with the smallest index
		 */
		// return Arrays.asList(items[argmax[0]]);
	}

	public <T> int[] getScores(Profile<T> profile, T[] items) {
		int numItem = items.length;
		int[] scores = new int[numItem];
		int index = -1;
		for (T[] preference : profile.data) {
			index++;
			for (int i = 0; i < numItem; i++) {
				int weight = numItem - i - 1;
				int k = Arrays.binarySearch(items, preference[i]);
				scores[k] += weight * profile.votes[index];
			}
		}
		return scores;
	}
}