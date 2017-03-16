package com.horsehour.vote.rule;

import java.util.Arrays;

import com.horsehour.vote.Profile;

/**
 * Veto rule is very easy to understand, where the item with least vetoes, or
 * receives the fewest last choice in voting preferences win
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:42:35 AM, Jun 19, 2016
 *
 */

public class Veto extends PositionalVotingRule {
	@Override
	protected double[] getPositionalScores(int length) {
		double[] scores = new double[length];
		Arrays.fill(scores, 1);
		scores[length - 1] = 0;
		return scores;
	}

	public static void main(String[] args) {
		String[][] preferences = { { "Memphis", "Nashville", "Chattanooga", "Knoxville" },
				{ "Nashville", "Chattanooga", "Knoxville", "Memphis" },
				{ "Chattanooga", "Knoxville", "Nashville", "Memphis" },
				{ "Knoxville", "Chattanooga", "Nashville", "Memphis" } };

		int[] votes = { 42, 26, 15, 17 };

		Profile<String> profile = new Profile<>(preferences, votes);

		Veto rule = new Veto();

		System.out.println(rule.getAllWinners(profile));
	}
}
