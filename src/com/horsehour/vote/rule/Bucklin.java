package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.axiom.NeutralityCriterion;
import com.horsehour.vote.axiom.VotingAxiom;

/**
 * Bucklin voting is a class of voting systems that can be used for both
 * single-member and multi-member districts. It is named after its earliest
 * promoter, James W. Bucklin of Grand Junction, Colorado, and is also known as
 * the Grand Junction system.
 * <p>
 * As in Majority Judgment, the Bucklin winner will be one of the candidates
 * with the highest median ranking or rating. Voters are allowed rank preference
 * ballots (first, second, third, etc.). First choice votes are first counted.
 * If one candidate has a majority, that candidate wins. Otherwise the second
 * choices are added to the first choices. Again, if a candidate with a majority
 * vote is found, the winner is the candidate with the most votes accumulated.
 * Lower rankings are added as needed.
 * <p>
 * A majority is determined based on the number of valid ballots. Since, after
 * the first round, there may be more votes cast than voters, it is possible for
 * more than one candidate to have majority support.
 * <p>
 * In the United States in the early-20th-century Progressive era, various
 * municipalities began to use Bucklin voting. It is no longer used in any
 * government elections, and has even been declared unconstitutional in
 * Minnesota.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 4:45:32 PM, Jun 20, 2016
 *
 */

public class Bucklin extends VotingRule {

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		return getAllWinners(profile);
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] scores = new int[items.length];
		cumulateScore(scores, 0, items, profile);
		int[] argmax = MathLib.argmax(scores);

		/**
		 * select all candidates with the largest scores
		 */
		List<T> winners = new ArrayList<>();
		for (int i : argmax)
			winners.add(items[i]);
		return winners;
	}

	<T> void cumulateScore(int[] votes, int rank, T[] items, Profile<T> profile) {
		if (rank == items.length)// unable to find a majority winner
			return;

		for (int i = 0; i < profile.data.length; i++) {
			int index = Arrays.binarySearch(items, profile.data[i][rank]);
			votes[index] += profile.votes[i];
		}

		int[] argmax = MathLib.argmax(votes);
		int maxVote = votes[argmax[0]];
		if (maxVote <= profile.numVoteTotal * 1.0 / 2)
			cumulateScore(votes, rank + 1, items, profile);
		else
			return;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		VotingRule rule = null;
		rule = new Bucklin();

		int numItem = 3, numVote = 5;

		VotingAxiom axiom = null;
		// axiom = new CondorcetCriterion();

		axiom = new NeutralityCriterion();

		System.out.println(axiom.getSatisfiability(numItem, numVote, rule));

		TickClock.stopTick();
	}
}