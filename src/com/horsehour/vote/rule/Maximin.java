package com.horsehour.vote.rule;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;

/**
 * Maximin (a.k.a Simpson-Krammer) is a Condorcet consistent rule
 * 
 * @version 1.0
 */
public class Maximin extends ScoredVotingRule {

	@Override
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] scores = getScores(profile, items);
		return new ScoredItems<>(items, scores);
	}

	/**
	 * @param profile
	 * @param sortedItems
	 * @return minimum pairwise preferred votes
	 */
	<T> int[] getScores(Profile<T> profile, T[] sortedItems) {
		int[][] ppm = Condorcet.getPairwisePreferenceMatrix(profile, sortedItems);
		
		int m = ppm.length;
		int[] scores = new int[m];

		for (int i = 0; i < m; i++) {
			scores[i] = Integer.MAX_VALUE;
			for (int j = 0; j < m; j++) {
				if (i == j)
					continue;
				if (ppm[i][j] < scores[i])
					scores[i] = ppm[i][j];
			}
		}
		return scores;
	}
}
