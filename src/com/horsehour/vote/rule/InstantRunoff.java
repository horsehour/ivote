package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.train.Learner;

/**
 * Instant Run-off Voting is aka the alternative vote (AV), Hare method,
 * transferable vote, (single-seat) ranked choice voting (RCV), preferential
 * voting, or plurality with elimination. It's a special case of Single
 * Transferable Voting (STV) rule for single-seat election. The voting rule
 * iteratively eliminates candidates who have least first place votes until some
 * one has majority of first place votes.
 * <p>
 * In Australia, IRV was first adopted in 1893, and continues to be used along
 * with STV today. The method was also used to select winner of Oscars.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:59:18 PM, Jun 18, 2016
 *
 */

public class InstantRunoff extends VotingRule {
	public boolean singleElimination = true;

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		return getAllWinners(profile);
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		return Arrays.asList(getSingleWinner(profile));
	}

	public <T> T getSingleWinner(Profile<T> profile) {
		// Arrays.asList() returns a fixed-length, immutable list
		List<T> itemList = new ArrayList<>(Arrays.asList(profile.getSortedItems()));
		List<List<T>> preferences = new ArrayList<>(profile.votes.length);

		for (T[] pref : profile.data)
			preferences.add(new ArrayList<>(Arrays.asList(pref)));

		List<Integer> votes = new ArrayList<>();
		for (int k : profile.votes)
			votes.add(k);

		int majVote = 1 + profile.numVoteTotal / 2;
		return runoff(preferences, votes, itemList, majVote);
	}

	/**
	 * 
	 * @param preferences
	 * @param votes
	 * @param items
	 * @param majVote
	 * @return the winner after a run-off election
	 */
	<T> T runoff(List<List<T>> preferences, List<Integer> votes, List<T> items, int majVote) {
		int numItem = items.size();

		if (numItem == 1)
			return items.get(0);

		int[] numFirstChoices = new int[numItem];
		List<T> pref = null;
		for (int i = 0; i < preferences.size(); i++) {
			pref = preferences.get(i);
			int index = items.indexOf(pref.get(0));
			numFirstChoices[index] += votes.get(i); // first choice
		}

		int count = 0;
		for (int nFirstChoice : numFirstChoices) {
			if (nFirstChoice >= majVote)
				return items.get(count);
			count++;
		}

		int[] argmin = MathLib.argmin(numFirstChoices);

		if (singleElimination)
			eliminate(preferences, items, argmin[0]);
		else {
			for (int i = argmin.length - 1; i >= 0; i--) {
				int index = argmin[i];
				eliminate(preferences, items, index);
			}
		}
		return runoff(preferences, votes, items, majVote);
	}

	/**
	 * Eliminate an item from the preference profile each time
	 *
	 * @param preferenceList
	 * @param itemList
	 * @param index
	 *            index of item which will be eliminated from profile
	 */
	<T> void eliminate(List<List<T>> preferenceList, List<T> itemList, int index) {
		T item = itemList.remove(index);
		for (List<T> pref : preferenceList)
			pref.remove(item);
	}

	public String toString() {
		return "IRV" + (singleElimination ? "-SE" : "-ME");
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String[][] preferences = { { "a", "b", "c" }, { "c", "a", "b" }, { "b", "c", "a" } };

		int[] votes = { 27, 42, 24 };

		// String[][] preferences = { { "Memphis", "Nashville", "Chattanooga",
		// "Knoxville" },
		// { "Nashville", "Chattanooga", "Knoxville", "Memphis" },
		// { "Chattanooga", "Knoxville", "Nashville", "Memphis" },
		// { "Knoxville", "Chattanooga", "Nashville", "Memphis" } };
		//
		// int[] votes = { 42, 26, 15, 17 };

		Profile<String> profile = new Profile<>(preferences, votes);

		List<VotingRule> rules = Arrays.asList(new Borda(), new Condorcet(), new Coombs(), new Copeland(),
				new InstantRunoff(), new KemenyYoung(), new Maximin(), new PairMargin(), new Plurality(),
				new RankedPairs(), new Schulze(), new Veto());

		rules.stream().count();
		VotingRule rule = new Learner().getEnsembleRule(rules);

		StringBuffer sb = new StringBuffer();
		for (VotingRule r : rules)
			sb.append(r.getAllWinners(profile) + " - " + r.toString() + "\n");
		sb.append(rule.getAllWinners(profile) + " - EnsembleRule\n");

		System.out.println(sb.toString());

		TickClock.stopTick();
	}
}
