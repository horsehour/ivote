package com.horsehour.vote.rule;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Rated ballots which allow equal or skipped rankings are cast by voters.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 1:20:32 AM, Aug 12, 2016
 */
public abstract class RatedVotingRule {
	/**
	 * The first level is the lowest and the last one the highest level
	 */
	public int[] levels;
	public int nLevel;

	public RatedVotingRule(int nLevel) {
		this.nLevel = nLevel;
		this.levels = new int[nLevel];
		for (int i = 0; i < nLevel; i++)
			levels[i] = i;
	}

	public RatedVotingRule(int[] levels) {
		this.levels = levels;
		this.nLevel = levels.length;
	}

	public abstract <T> List<T> getAllWinners(T[] items, int[][] preferences, int[] votes);

	/**
	 * 
	 * @param items
	 * @param preferences
	 *            Preferences with default one vote each preference
	 * @return candidates with the highest approves win
	 */
	public <T> List<T> getAllWinners(T[] items, int[][] preferences) {
		Objects.requireNonNull(items, "Items should not be null.");
		Objects.requireNonNull(preferences, "Preferences should not be null.");

		int[] votes = new int[preferences.length];
		Arrays.fill(votes, 1);
		return getAllWinners(items, preferences, votes);
	}

	public String getName() {
		return getClass().getSimpleName();
	}
}