package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.horsehour.util.MathLib;

/**
 * Majority Judgment is a single-winner voting system proposed by Michel
 * Balinski and Rida Laraki. Voters freely grade each candidate in one of
 * several named ranks, for instance from "excellent" to "bad", and the
 * candidate with the highest median grade is the winner. If more than one
 * candidate has the same median grade, a tiebreaker is used which sees the
 * "closest-to-median" grade.
 * <p>
 * Majority Judgment can be considered as a form of Bucklin voting which allows
 * equal ranks. Voters are allowed rated ballots, on which they may assign a
 * grade or judgment to each candidate. Badinski and Laraki suggest six grading
 * levels, from "Excellent" to "Reject". Multiple candidates may be given the
 * same grade if the voter desires. The median grade for each candidate is
 * found, for instance by sorting their list of grades and finding the middle
 * one. If the middle falls between two different grades, the lower of the two
 * is used. The candidate with the highest median grade wins. If several
 * candidates share the highest median grade, all other candidates are
 * eliminated. Then, one copy of that grade is removed from each remaining
 * candidate's list of grades, and the new median is found, until there is an
 * unambiguous winner. For instance, if candidate X's sorted ratings were
 * {"Good", "Good", "Fair", "Poor"}, while candidate Y had {"Excellent", "Fair",
 * "Fair", "Fair"}, the rounded medians would both be "Fair". After removing one
 * "Fair" from each list, the new lists are, respectively, {"Good", "Good",
 * "Poor"} and {"Excellent", "Fair", "Fair"}, so X would win with a recalculated
 * median of "Good".
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 4:45:32 PM, Jun 20, 2016
 *
 */

public class MajorityJudgement extends RatedVotingRule {

	public MajorityJudgement(int[] levels) {
		super(levels);
	}

	public MajorityJudgement(int nLevel) {
		super(nLevel);
	}

	/**
	 * @param items
	 * @param preferences
	 * @return candidates with the highest median rate win
	 */
	public <T> List<T> getAllWinners(T[] items, int[][] preferences, int[] votes) {
		Objects.requireNonNull(items, "Items should not be null.");
		Objects.requireNonNull(preferences, "Preferences should not be null.");
		Objects.requireNonNull(votes, "Votes should not be null.");

		if (preferences.length != votes.length) {
			System.err.println("Preferences and votes should have the same dimension.");
			return null;
		}

		// collect rates
		List<List<Integer>> rates = new ArrayList<>(items.length);
		for (int i = 0; i < items.length; i++)
			rates.add(new ArrayList<>());

		int index = -1;
		for (int[] preference : preferences) {
			index++;
			for (int i = 0; i < preference.length; i++)
				for (int m = 0; m < votes[index]; m++)
					rates.get(i).add(preference[i]);
		}

		for (int i = 0; i < items.length; i++)
			Collections.sort(rates.get(i));

		int[] indices = new int[items.length];
		for (int i = 0; i < items.length; i++)
			indices[i] = i;
		T winner = getHighestMedianItem(rates, indices, items);
		return Arrays.asList(winner);
	}

	<T> T getHighestMedianItem(List<List<Integer>> rates, int[] indices, T[] items) {
		List<Integer> medians = new ArrayList<>();
		for (int index : indices) {
			List<Integer> rateList = rates.get(index);
			int size = rateList.size();
			if (size % 2 == 0)
				medians.add(rateList.get(size / 2 - 1));
			else
				medians.add(rateList.get(size / 2));
		}

		int[] argmax = MathLib.argmax(medians);
		if (argmax.length == 1)
			return items[indices[argmax[0]]];
		else {
			Integer highestMedian = medians.get(argmax[0]);
			// eliminate one highest median rate from each remaining candidates
			int[] nextIndices = new int[argmax.length];
			for (int i = 0; i < argmax.length; i++) {
				int index = indices[argmax[i]];
				nextIndices[i] = index;
				rates.get(index).remove(highestMedian);
			}
			return getHighestMedianItem(rates, nextIndices, items);
		}
	}

	public static void main(String[] args) {
		String[] items = { "Memphis", "Nashville", "Chattanooga", "Knoxville" };

		int[][] preferences = { { 4, 2, 1, 1 }, { 1, 4, 2, 2 }, { 1, 2, 4, 3 }, { 1, 2, 3, 4 } };
		int[] votes = { 42, 26, 15, 17 };

		MajorityJudgement rule = new MajorityJudgement(new int[] { 1, 2, 3, 4 });
		List<String> winners = rule.getAllWinners(items, preferences, votes);
		System.out.println(winners.toString());
	}
}