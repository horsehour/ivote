package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;

/**
 * The voting rule was devised by Joseph M. Baldwin. It is essentially Borda
 * count combined with the instant-runoff voting procedure. First, the Borda
 * scores of all candidates are computed, and then the candidate with the lowest
 * score is eliminated. Then the Borda scores of each remaining candidate are
 * recomputed, as if the eliminated candidate were not on the ballot. This is
 * repeated until there is a final candidate left.
 * <p>
 * The Baldwin method satisfy the Condorcet criterion: since Borda always gives
 * any existing Condorcet winner more than the average Borda points, the
 * Condorcet winner will never be eliminated. It does not satisfy the
 * independence of irrelevant alternatives criterion, the monotonicity
 * criterion, the participation criterion, the consistency criterion and the
 * independence of clones criterion, while it does satisfy the majority
 * criterion, the mutual majority criterion, the Condorcet loser criterion, and
 * the Smith criterion. Also, the Baldwin method violates reversal symmetry.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:06:54 PM, Jul 31, 2016
 *
 */

public class Baldwin extends InstantRunoff {
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

		int[] argmin = MathLib.argmin(scores);
		if (argmin.length == numItem)
			return items;

		for (int i = argmin.length - 1; i >= 0; i--) {
			int index = argmin[i];
			eliminate(preferences, items, index);
		}
		return runoff(preferences, votes, items);
	}

	public String toString() {
		return getClass().getSimpleName();
	}
}
