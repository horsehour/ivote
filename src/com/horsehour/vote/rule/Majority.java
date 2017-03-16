package com.horsehour.vote.rule;

import java.util.Arrays;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;

/**
 * Majority rule is a decision rule that selects alternatives which have a
 * majority, that is, more than half the votes.
 * <p>
 * Though plurality (first-past-the post) is often mistaken for majority rule,
 * they are not the same. Plurality makes the options with the most votes the
 * winner, regardless of whether the fifty percent threshold is passed. This is
 * equivalent to majority rule when there are only two alternatives. However,
 * when there are more than two alternatives, it is possible for plurality to
 * choose an alternative that has fewer than fifty percent of the votes cast in
 * its favor.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:21:16 PM, Jul 31, 2016
 *
 */

public class Majority extends VotingRule {

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		return getAllWinners(profile);
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		T winner = getWinner(profile);
		if(winner == null)
			return null;
		else
			return Arrays.asList(winner);
	}
	
	public <T> T getWinner(Profile<T> profile){
		T[] items = profile.getSortedItems();
		int[] scores = new int[items.length];
		int index = 0;
		for (T[] preference : profile.data) {
			int i = Arrays.binarySearch(items, preference[0]);
			scores[i] += profile.votes[index++];
		}

		int[] argmax = MathLib.argmax(scores);
		T winner = items[argmax[0]];
		if(scores[argmax[0]] > profile.numVoteTotal * 1.0 / 2)
			return winner;
		else
			return null;
	}
	
	public <T> T getLoser(Profile<T> profile){
		T[] items = profile.getSortedItems();
		int[] scores = new int[items.length];
		int index = 0, indexLast = items.length - 1;
		for (T[] preference : profile.data) {
			int i = Arrays.binarySearch(items, preference[indexLast]);
			scores[i] += profile.votes[index++];
		}

		int[] argmax = MathLib.argmax(scores);
		T loser = items[argmax[0]];
		if(scores[argmax[0]] > profile.numVoteTotal * 1.0 / 2)
			return loser;
		else
			return null;
	}
}
