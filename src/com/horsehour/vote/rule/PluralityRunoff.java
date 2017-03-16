package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;

/**
 * Plurality with Run-off rule is a 2-round election method. In the 1st round,
 * two alternatives with highest plurality scores survive. In the 2nd round, the
 * winner of a pairwise election between the two is the final winner. If two
 * candidates have the same number of first-place votes, then check their number
 * of second-place votes and eliminate the candidate with fewer second-place
 * votes.
 * <p><b>Applications:</b> be used in French president election.
 * <p><b>Axiomatic Properties:</b> violates monotonicity criterion
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 4:18:43 AM, Jun 19, 2016
 *
 */

public class PluralityRunoff extends VotingRule {
	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		return null;
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		return Arrays.asList(getSingleWinner(profile));
	}

	public <T> T getSingleWinner(Profile<T> profile) {
		T[] sortedItems = profile.getSortedItems();
		int numItem = sortedItems.length;
		
		List<Integer> tieIndices = IntStream.range(0, numItem).boxed().collect(Collectors.toList());
		List<Integer> duoPlayer = pluralitySelect(profile, sortedItems, tieIndices, 0, 2);

		int i = 0, winnings = 0;
		for (T[] pref : profile.data) {
			int r1 = ArrayUtils.indexOf(pref, sortedItems[duoPlayer.get(0)]);
			int r2 = ArrayUtils.indexOf(pref, sortedItems[duoPlayer.get(1)]);
			if (r1 < r2)
				winnings += profile.votes[i];
			i++;
		}

		int margin = 2 * winnings - profile.getNumVote();
		T winner = null;
		if (margin > 0)
			winner = sortedItems[duoPlayer.get(0)];
		else if (margin < 0)
			winner = sortedItems[duoPlayer.get(1)];

		// if margin == 0, it implies that there is no winner at all
		return winner;

	}

	/**
	 * When selecting top two using plurality procedure. Two kinds of ties may
	 * happen. The first kind of ties exists if there are more than two
	 * alternatives who have the most first place votes. We have to refer to
	 * their second-place votes for tie-breaking. When they have the same
	 * second-place votes, we then count their third-place votes to break the
	 * tie, until we could find the duo-player. As for the second type of ties,
	 * it takes place when more than two alternatives are runner-ups. We have to
	 * repeatedly compute their k-th places votes to select an unique runner-up.
	 * 
	 * @param profile
	 * @param sortedItems
	 * @param tieIndices
	 *            items who are tied
	 * @param place
	 *            the place for votes counting
	 * @param numSel
	 *            number of items to be selected from tieItems, 1 or 2
	 * @return selected index (or indices) of item(s) for the later head-to-head
	 *         competition
	 */
	<T> List<Integer> pluralitySelect(Profile<T> profile, T[] sortedItems, List<Integer> tieIndices, int place,
			int numSel) {
		// no need for tie-breaking
		if (numSel == tieIndices.size() || tieIndices.size() == 1)
			return tieIndices;

		int size = tieIndices.size();
		int[] scores = new int[size];
		for (int i = 0; i < size; i++) {
			int id = tieIndices.get(i);
			int vid = 0;
			for (T[] pref : profile.data) {
				if (pref[place].equals(sortedItems[id]))
					scores[i] += profile.votes[vid];
				vid++;
			}
		}

		int[] indices = IntStream.range(0, size).boxed()
				.sorted((i, j) -> Integer.valueOf(scores[j]).compareTo(Integer.valueOf(scores[i])))
				.mapToInt(i -> i).toArray();

		List<Integer> ret = new ArrayList<>();
		List<Integer> newTieIndices = new ArrayList<>();
		if (scores[indices[0]] == scores[indices[1]]) {
			for (int i = 0; i < size; i++)
				if (scores[indices[0]] == scores[indices[i]])
					newTieIndices.add(tieIndices.get(indices[i]));
				else
					break;
			ret = pluralitySelect(profile, sortedItems, newTieIndices, place + 1, numSel);
		} else if (scores[indices[0]] > scores[indices[1]]) {
			ret.add(tieIndices.get(indices[0]));
			newTieIndices.add(tieIndices.get(indices[1]));
			for (int i = 2; i < size; i++)
				if (scores[indices[1]] == scores[indices[i]])
					newTieIndices.add(tieIndices.get(indices[i]));
				else
					break;
			ret.addAll(pluralitySelect(profile, sortedItems, newTieIndices, place + 1, numSel - 1));
		}
		return ret;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String[][] preferences = { { "a", "b", "c", "d", "e" }, { "b", "d", "c", "e", "a" },
				{ "c", "d", "b", "a", "e" }, { "c", "e", "b", "d", "a" }, { "d", "e", "c", "b", "a" },
				{ "e", "c", "b", "d", "a" } };
		int[] multiplicities = { 33, 16, 3, 8, 18, 22 };

//		String[][] preferences = { { "Memphis", "Nashville", "Chattanooga", "Knoxville" },
//				{ "Nashville", "Chattanooga", "Knoxville", "Memphis" },
//				{ "Chattanooga", "Knoxville", "Nashville", "Memphis" },
//				{ "Knoxville", "Chattanooga", "Nashville", "Memphis" } };
//
//		int[] multiplicities = { 42, 26, 15, 17 };

		Profile<String> profile = new Profile<>(preferences, multiplicities);

		PluralityRunoff rule = new PluralityRunoff();
		// rule.singleElimination = true;

		System.out.println(rule.getSingleWinner(profile));

		TickClock.stopTick();
	}
}
