package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;

/**
 * Based on the work of the mathematician Edward J. Nanson, Nanson's method
 * eliminates those choices from a Borda count tally that are at or below the
 * average Borda count score, then the ballots are retallied as if the remaining
 * candidates were exclusively on the ballot. This process is repeated until a
 * single winner remains.
 * <p>
 * The Nanson method and the Baldwin method satisfy the Condorcet criterion:
 * since Borda always gives any existing Condorcet winner more than the average
 * Borda points, the Condorcet winner will never be eliminated. They do not
 * satisfy the independence of irrelevant alternatives criterion, the
 * monotonicity criterion, the participation criterion, the consistency
 * criterion and the independence of clones criterion, while they do satisfy the
 * majority criterion, the mutual majority criterion, the Condorcet loser
 * criterion, and the Smith criterion. The Nanson method satisfies reversal
 * symmetry, while the Baldwin method violates reversal symmetry.
 * <p>
 * Nanson's method was used in city elections in the U.S. town of Marquette,
 * Michigan in the 1920s. It was formally used by the Anglican Diocese of
 * Melbourne and in the election of members of the University Council of the
 * University of Adelaide. It was used by the University of Melbourne until
 * 1983.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:06:54 PM, Jul 31, 2016
 *
 */

public class Nanson extends InstantRunoff {
	public <T> List<T> getAllWinners(Profile<T> profile) {
		// Arrays.asList() returns a fixed-length, immutable list
		List<T> itemList = new ArrayList<>(Arrays.asList(profile.getSortedItems()));
		List<List<T>> preferences = new ArrayList<>(profile.votes.length);

		for (T[] pref : profile.data)
			preferences.add(new ArrayList<>(Arrays.asList(pref)));

		List<Integer> votes = new ArrayList<>();
		for (int k : profile.votes)
			votes.add(k);
		return runoff(preferences, votes, itemList);
	}

	/**
	 * 
	 * @param preferences
	 * @param votes
	 * @param items
	 * @param majVote
	 * @return the winner after a run-off election
	 */
	<T> List<T> runoff(List<List<T>> preferences, List<Integer> votes, List<T> items) {
		int numItem = items.size();

		if (numItem == 1)
			return items;

		int[] scores = new int[numItem];
		List<T> pref = null;
		for (int i = 0; i < preferences.size(); i++) {
			pref = preferences.get(i);
			for (int k = 0; k < numItem; k++) {
				int index = items.indexOf(pref.get(k));
				scores[index] += votes.get(i) * (numItem - k - 1);
			}
		}

		double average = MathLib.Data.mean(scores);

		for (int i = numItem - 1; i >= 0; i--) {
			if (scores[i] < average)
				eliminate(preferences, items, i);
		}

		/**
		 * no elimination, tied winners
		 */
		if (items.size() == numItem)
			return items;

		return runoff(preferences, votes, items);
	}

	public String toString() {
		return getClass().getSimpleName();
	}
}
