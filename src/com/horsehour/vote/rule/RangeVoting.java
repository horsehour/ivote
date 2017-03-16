package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Range voting is a voting system for one-seat elections under which voters
 * score each candidate, the scores are added up, and the candidate with the
 * highest score wins. Range voting is also known ratings summation, average
 * voting, cardinal ratings, score voting, 0–99 voting, the score system, or the
 * point system.
 * <p>
 * A form of range voting was apparently used in some elections in Ancient
 * Sparta by measuring how loudly the crowd shouted for different candidates;
 * rough modern-day equivalents include the use of clapometers in some
 * television shows and the judging processes of some athletic competitions.
 * <p>
 * Range voting satisfies the monotonicity criterion, i.e. raising your vote's
 * score for a candidate can never hurt their chances of winning. Also, in range
 * voting, casting a sincere vote can never result in a worse election winner
 * (from your point of view) than if you had simply abstained from voting. Range
 * voting passes the favorite betrayal criterion, meaning that it never gives
 * voters an incentive to rate their favorite candidate lower than a candidate
 * they like less.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 1:13:11 AM, Jun 20, 2016
 *
 */

public class RangeVoting {
	/**
	 * @param items
	 * @param scoreVotes
	 *            ranged values are allowed, if left with -1, means that the
	 *            voter does not cast a vote to the corresponding item
	 * @return all items who get the largest average number of votes
	 */
	public <T> List<T> getAllWinners(T[] items, int[][] scoreVotes) {
		double[] avgScore = getAverageScores(items, scoreVotes);
		int[] indices = IntStream.range(0, items.length).boxed()
				.sorted((i, j) -> Double.compare(avgScore[j], avgScore[i])).mapToInt(i -> i).toArray();
		double highestScore = avgScore[indices[0]];
		int i = 1;
		for (; i < items.length; i++)
			if (highestScore > avgScore[indices[i]])
				break;

		List<T> winnerList = new ArrayList<>();
		for (int k = 0; k < i; k++)
			winnerList.add(items[indices[k]]);
		return winnerList;
	}

	/**
	 * Calculate the average score of each item
	 * 
	 * @param items
	 * @param scoreVotes
	 * @return average scores of items
	 */
	<T> double[] getAverageScores(T[] items, int[][] scoreVotes) {
		int numItem = items.length;
		double[] ret = new double[numItem];
		for (int i = 0; i < numItem; i++) {
			int numVotes = 0;
			for (int[] scores : scoreVotes)
				if (scores[i] > -1) {
					ret[i] += scores[i];
					numVotes++;
				}
			ret[i] /= numVotes;
		}
		return ret;
	}

	public String getName() {
		return getClass().getSimpleName();
	}

	public static void main(String[] args) {
		Integer[] items = { 1, 2, 3 };
		int[][] scoreVotes = { { 90, 20, 0 }, { 10, 90, -1 }, { 20, 10, -1 } };

		RangeVoting rule = new RangeVoting();
		List<Integer> winners = rule.getAllWinners(items, scoreVotes);
		System.out.println(winners.toString());
	}
}
