package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.train.Eval;

/**
 * The Oklahoma primary electoral system was a voting system used to elect one
 * winner from a pool of candidates using preferential voting. Voters rank
 * candidates in order of preference (any voter who did not rank enough
 * candidates would have the ballot voided), and their votes are initially
 * allocated to their first-choice candidate. If, after this initial count, no
 * candidate has a majority of votes cast, a mathematical formula comes into
 * play. The system was used for primary elections in Oklahoma when it was
 * adopted in 1925 until it was ruled unconstitutional by the Supreme Court of
 * Oklahoma in 1926.
 * <p>
 * In the event that no single person received a majority of the
 * first-preference votes, every candidate would have half the number of
 * second-preference votes added to their total. If, after this, any candidate
 * who had a majority of votes cast would be declared winner; if not, and there
 * were only two preferences expressed, the winner would be whoever had the
 * higher total. If, however, there were five or more candidates and none held a
 * majority after this second round, then each would have one-third of the
 * third-preference votes added on, and after this, whoever had the highest
 * total would be declared winner.
 * <p>
 * The method requires voters to rank their preferences, which contrasted with
 * most other states' procedures merely giving people the option of doing so
 * (for that matter, only eight states used preferential voting at all),[5] was
 * an attempt to balance the competing concerns of preventing <b>bullet
 * voting</b> (people deciding to list only their first choice).
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 4:45:32 PM, Jun 20, 2016
 *
 */

public class OklahomaVoting extends VotingRule {
	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();

		int numItem = items.length, position = 0;
		float majVote = profile.numVoteTotal * 1.0F / 2;

		float[] scores = new float[numItem];
		List<T> ranking = new ArrayList<>();
		while (ranking.isEmpty() && position < numItem) {
			getScores(profile, items, scores, position++);
			for (int i = 0; i < numItem; i++) {
				if (scores[i] > majVote)
					ranking.add(items[i]);
			}
		}

		if (!ranking.isEmpty())
			ranking = new ArrayList<>();
		int[] rank = MathLib.getRank(scores, false);
		for (int i : rank)
			ranking.add(items[i]);
		return ranking;
	}

	public <T> List<T> getAllWinners(Profile<T> profile) {
		T[] items = profile.getSortedItems();

		int numItem = items.length, position = 0;
		float majVote = profile.numVoteTotal * 1.0F / 2;

		float[] scores = new float[numItem];
		List<T> winners = new ArrayList<>();
		while (winners.isEmpty() && position < numItem) {
			getScores(profile, items, scores, position++);
			for (int i = 0; i < numItem; i++) {
				if (scores[i] > majVote)
					winners.add(items[i]);
			}
		}

		if (winners.isEmpty()) {
			int[] argmax = MathLib.argmax(scores);
			for (int i : argmax)
				winners.add(items[i]);
		}

		return winners;
	}

	<T> float[] getScores(Profile<T> profile, T[] sortedItems, float[] scores, int position) {
		for (int i = 0; i < profile.data.length; i++) {
			T[] preferences = profile.data[i];
			int index = Arrays.binarySearch(sortedItems, preferences[position]);
			scores[index] += profile.votes[i] * 1.0F / (position + 1);
		}
		return scores;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		Eval eval = new Eval();
		System.out.println(eval.getSimilarity(3, 11, new Borda(), new OklahomaVoting()));

		TickClock.stopTick();
	}
}