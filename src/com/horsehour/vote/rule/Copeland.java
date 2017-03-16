package com.horsehour.vote.rule;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 9:29:39 PM, Jun 18, 2016
 *
 */

public class Copeland extends ScoredVotingRule {

	@Override
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] scores = getWinningMinusLoseSocre(profile, items);
		return new ScoredItems<T>(items, scores);
	}

	<T> int[] getWinningMinusLoseSocre(Profile<T> profile, T[] sortedItems) {
		int[][] ppm = Condorcet.getPairwisePreferenceMatrix(profile, sortedItems);
		int numItem = profile.getNumItem();
		int[] score = new int[numItem];
		for (int i = 0; i < numItem; i++) {
			int numWinning = 0, numLoss = 0;
			for (int j = 0; j < numItem; j++) {
				if (i == j)
					continue;

				if (ppm[i][j] > ppm[j][i])
					numWinning++;
				else if (ppm[i][j] < ppm[j][i])
					numLoss++;
			}
			score[i] = numWinning - numLoss;
		}
		return score;
	}

	public static void main(String[] args) {
		Integer[][] preferences = { { 1, 2, 3, 4 }, { 2, 1, 3, 4 }, { 1, 2, 4, 3 } };
		Profile<Integer> profile = new Profile<>(preferences);

		Copeland rule = new Copeland();
		System.out.println(rule.getAllWinners(profile));
	}
}
