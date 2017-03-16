package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import com.horsehour.util.MathLib;

/**
 * Approval voting is a single-winner voting system used for elections. The
 * voting rule allows every individual votes for as many alternatives as s/he
 * wishes to and all alternatives with the highest number of votes are elected.
 * <p>
 * The system was described in 1976 by Guy Ottewell and also by Robert J. Weber,
 * who coined the term "approval voting." It was more fully published in 1978 by
 * political scientist Steven Brams and mathematician Peter Fishburn.
 * <p>
 * Approval voting can be considered as a form of range voting, with the range
 * restricted to two values, 0 and 1, or a form of Majority Judgment, with the
 * grades restricted to "Good" and "Poor". Approval voting can also be compared
 * to plurality voting, without the rule that discards ballots which vote for
 * more than one candidate. Ballots which mark every candidate the same (whether
 * yes or no) have no effect on the outcome of the election. Each ballot can
 * therefore be viewed as a small "delta" which separates two groups of
 * candidates, or a single-pair of ranks (e.g. if a ballot indicates that A & C
 * are approved and B & D are not, the ballot can be considered to convey the
 * ranking [A=C]>[B=D]).
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 4:45:32 PM, Jun 20, 2016
 *
 */

public class ApprovalVoting extends RatedVotingRule{
	public ApprovalVoting(int nLevel) {
		super(nLevel);
	}

	public ApprovalVoting(int[] levels) {
		super(levels);
	}

	/**
	 * 
	 * @param items
	 * @param preferences
	 * @param votes
	 * @return candidates with the highest number of approves win
	 */
	public <T> List<T> getAllWinners(T[] items, int[][] preferences, int[] votes) {
		Objects.requireNonNull(items, "Items should not be null.");
		Objects.requireNonNull(preferences, "Preferences should not be null.");
		Objects.requireNonNull(votes, "Votes should not be null.");

		if (preferences.length != votes.length) {
			System.err.println("Preferences and votes should have the same dimension.");
			return null;
		}

		// only the highest level be used to tally approves
		int approveLevel = levels[nLevel - 1];
		int[] approves = new int[items.length];

		// count approves
		int index = -1;
		for (int[] preference : preferences) {
			index++;
			for (int i = 0; i < preference.length; i++) {
				if (preference[i] == approveLevel)
					approves[i] += votes[index];
			}
		}

		List<T> winners = new ArrayList<>();
		int[] indices = MathLib.argmax(approves);
		for (int i : indices)
			winners.add(items[i]);
		return winners;
	}

	public static void main(String[] args) {
		Integer[] items = { 1, 2, 3 };
		int[][] preferences = { { 1, 1, 0 }, { 1, 1, 0 }, { 1, 0, 1 }, { 0, 0, 1 } };

		ApprovalVoting rule = new ApprovalVoting(2);
		List<Integer> winners = rule.getAllWinners(items, preferences);
		System.out.println(winners.toString());
	}
}