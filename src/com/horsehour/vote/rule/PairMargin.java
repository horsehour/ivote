package com.horsehour.vote.rule;

import java.util.Arrays;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 7:07:02 PM, Jun 18, 2016
 *
 */

public class PairMargin extends ScoredVotingRule {
	@Override
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		double[] scores = getPairMarginScores(profile, items);
		return new ScoredItems<>(items, scores);
	}

	<T> double[] getPairMarginScores(Profile<T> profile, T[] items) {
		int numItem = items.length;

		double[][] marginTable = new double[numItem][numItem];
		int index = -1;
		for (T[] pref : profile.data) {
			index++;
			for (int i = 0; i < numItem; i++) {
				int runner = Arrays.binarySearch(items, pref[i]);
				for (int j = i + 1; j < numItem; j++) {
					int opponent = Arrays.binarySearch(items, pref[j]);
					float val = (j - i) * (1.0f / (i + 1)) * profile.votes[index];
					marginTable[runner][opponent] += val;
					marginTable[opponent][runner] -= val;
				}
			}
		}
		double[] scores = new double[numItem];
		for (int i = 0; i < numItem; i++)
			scores[i] = Arrays.stream(marginTable[i]).reduce(0.0, Double::sum);
		return scores;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		Integer[][] preferences = { { 1, 2, 3, 4, 5, 6 } };

		Profile<Integer> profile = new Profile<>(preferences);

		PairMargin rule = new PairMargin();
		ScoredItems<Integer> items = rule.getScoredRanking(profile);
		for (int item : items.keySet())
			System.out.println(item + "," + items.get(item));

		TickClock.stopTick();
	}
}
