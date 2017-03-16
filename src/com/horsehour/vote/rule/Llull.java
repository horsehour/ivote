package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;

/**
 * Ramon Llull devised the earliest known Condorcet method in 1299.[1] His
 * method did not have voters express orders of preference; instead, it had a
 * round of voting for each of the possible pairings of candidates. (This was
 * more like the Robert's Rules method except it was analogous to a round-robin
 * tournament instead of a single-elimination tournament.) The winner was the
 * alternative that won the most pairings.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:17:09 PM, Aug 18, 2016
 *
 */

public class Llull extends Condorcet {
	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] wins = getWinningNum(getPairwisePreferenceMatrix(profile, items));
		int[] argmax = MathLib.argmax(wins);
		List<T> winners = new ArrayList<>();
		for (int i : argmax)
			winners.add(items[i]);
		return winners;
	}
	
	public <T> List<T> getAllLosers(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] wins = getWinningNum(getPairwisePreferenceMatrix(profile, items));
		int[] argmin = MathLib.argmin(wins);
		List<T> losers = new ArrayList<>();
		for (int i : argmin)
			losers.add(items[i]);
		return losers;
	}
}
