package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import com.horsehour.vote.Profile;

/**
 *
 * Coombs' Rule is a ranked voting method designed by Clyde Coombs for
 * single-winner elections. Similar to IRV, it's built upon item elimination and
 * redistribution of votes cast for the eliminated item until one has a majority
 * of votes. The difference is the elimination procedure: IRV eliminates items
 * based on the number of first choices, but Coombs makes the decision using
 * both the number of first choices and the number of last choices in preference
 * profile.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:35:57 AM, Jun 19, 2016
 *
 */

public class Coombs extends VotingRule {
	public boolean singleElimination = true;

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		return null;
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		return Arrays.asList(getSingleWinner(profile));
	}

	public <T> T getSingleWinner(Profile<T> profile) {
		// Arrays.asList() return a fixed-length list, which is not allowed to
		// change
		List<T> itemList = new ArrayList<>(Arrays.asList(profile.getSortedItems()));
		List<List<T>> preferenceList = new ArrayList<>();

		for (T[] pref : profile.data)
			preferenceList.add(new ArrayList<>(Arrays.asList(pref)));

		List<Integer> voteList = new ArrayList<>();
		for (int k : profile.votes)
			voteList.add(k);

		int majorityVotes = 1 + profile.numVoteTotal / 2;

		return runoff(preferenceList, voteList, itemList, majorityVotes);
	}

	/**
	 * 
	 * @param preferenceList
	 * @param multiplicityList
	 * @param itemList
	 * @param majVotes
	 * @return the winner after a run-off election
	 */
	<T> T runoff(List<List<T>> preferenceList, List<Integer> multiplicityList, List<T> itemList, int majVotes) {
		int numItem = itemList.size();

		if (numItem == 1)
			return itemList.get(0);

		// First and Last Choices
		int[][] choices = new int[2][numItem];
		List<T> pref = null;
		for (int i = 0; i < preferenceList.size(); i++) {
			pref = preferenceList.get(i);
			int index = itemList.indexOf(pref.get(0));
			choices[0][index] += multiplicityList.get(i); // first choice

			index = itemList.indexOf(pref.get(numItem - 1));
			choices[1][index] += multiplicityList.get(i);// last choice
		}

		int count = 0;
		for (int nFirstChoices : choices[0]) {
			if (nFirstChoices >= majVotes)
				return itemList.get(count);
			count++;
		}

		int[] indices = IntStream.range(0, numItem).boxed()
				.sorted((i, j) -> Integer.valueOf(choices[1][j]).compareTo(Integer.valueOf(choices[1][i])))
				.mapToInt(e -> e).toArray();

		eliminate(preferenceList, itemList, indices[0]);

		if (!singleElimination) {
			int i = 1;
			for (; i < numItem; i++) {
				if (indices[i] == indices[0])
					eliminate(preferenceList, itemList, indices[i]);
				else
					break;
			}
		}
		return runoff(preferenceList, multiplicityList, itemList, majVotes);
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
		String name = super.toString();
		return name + (singleElimination ? "-SE" : "-ME");
	}

	public static void main(String[] args) {
		String[][] preferences = { { "Memphis", "Nashville", "Chattanooga", "Knoxville" },
				{ "Nashville", "Chattanooga", "Knoxville", "Memphis" },
				{ "Chattanooga", "Knoxville", "Nashville", "Memphis" },
				{ "Knoxville", "Chattanooga", "Nashville", "Memphis" } };

		int[] votes = { 42, 26, 15, 17 };

		Profile<String> profile = new Profile<>(preferences, votes);

		Coombs rule = new Coombs();
		// rule.singleElimination = true;

		System.out.println(rule.getSingleWinner(profile));
	}
}
