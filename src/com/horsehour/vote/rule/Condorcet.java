package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;

/**
 * A Condorcet method is any election method that elects the candidate that
 * would win by majority rule in all pairings against the other candidates,
 * whenever one of the candidates has that property. A candidate with that
 * property is called a Condorcet winner. Voting methods that always elect the
 * Condorcet winner (when one exists) are the ones that satisfy the Condorcet
 * criterion.
 * <p>
 * It is named for the 18th-century French mathematician and philosopher Marie
 * Jean Antoine Nicolas Caritat, the Marquis de Condorcet, who championed such
 * outcomes.
 * <p>
 * A Condorcet winner doesn't always exist because majority preferences can be
 * like rock-paper-scissors: for each candidate, there can be another that is
 * preferred by some majority (this is known as Condorcet paradox).
 * <p>
 * Ramon Llull devised the earliest known Condorcet method in 1299.[1] His
 * method did not have voters express orders of preference; instead, it had a
 * round of voting for each of the possible pairings of candidates. (This was
 * more like the Robert's Rules method except it was analogous to a round-robin
 * tournament instead of a single-elimination tournament.) The winner was the
 * alternative that won the most pairings.
 * <p>
 * With the discovery in 2001 of his lost manuscripts, Ars notandi, Ars
 * eleccionis, and Alia ars eleccionis, Llull is given credit for discovering
 * the Borda count and Condorcet criterion, which Jean-Charles de Borda and
 * Nicolas de Condorcet independently discovered centuries later.The terms Llull
 * winner and Llull loser are ideas in contemporary voting systems studies that
 * are named in honor of Llull.
 * <p>
 * To elect 100% of the Condorcet winners, we need a Condorcet-completion rule.
 * The Pairwise- or Condorcet-completion rules all give the same result in most
 * elections. They differ only when there is no pair-wise winner due to a
 * voting-cycle such as C over B, B over D, D over C (C > B > D > C). Each
 * completion rule is a way to resolve a voting cycle, and they differ in their
 * abilities to elect the most central member of a voting cycle and to resist
 * manipulation.
 * <li>Duncan Black’s 1958 rule elects the Condorcet winner if 1 exists;
 * otherwise it elects the Borda winner. It is the best completion rule for
 * electing the “utility maximizing” option, if there is no manipulation.</li>
 * <li>A. H. Copeland’s 1950 rule gives a candidate 1 point for win¬ning a
 * pairwise contest against another candidate and -1 for losing. (In voting
 * cycles, Copeland often produces ties – so it does not “complete” Condorcet.)
 * </li>
 * <li>Mathematician Charles Ludwidge Dodgson (better known as author Lewis
 * Carroll) proposed in 1876 to elect the Condorcet winner or, in the event of a
 * cycle, the candidate who needs to change the fewest ballots to become the
 * Condorcet winner.</li>
 * <li>John Kemeny’s 1959 system determines how many rank pairs must be
 * exchanged (flipped) on voters’ ballots to make a candidate win by Condorcet’s
 * rule. The candidate who requires the fewest changes wins. The Kemeny distance
 * between two preference orders is the number of adjacent pairwise switches
 * needed to convert one preference order to the other.</li>
 * <li>The Minimax system elects the can¬didate with the smallest pairwise loss.
 * (It is not the same as Dodgson. A candidate may lose pairwise elections to
 * two rivals by 5% each. Her Minimax score would be -5%. But she might have to
 * change 10% of the ballots to become Dodgson’s winner.)</li>
 * <li>Markus Schulze's method takes the candidates in the voting cycle and
 * finds the stongest path from one candidate to another. For example, A>C>B is
 * a path for A over B, even if B topped A 1 against 1, (B>A). The strength of a
 * path is the strength of its weakest link. It elects the candidate who's
 * weakest path is stronger than every other candidate's weakest path.</li>
 * <li>Nicholas Tideman's Ranked Pairs rule creates a complete ranking of the
 * candidates from first to last. Their ranks come from majority preferences
 * between options: The biggest margin of victory in the Pairwise table is
 * locked in, say C > B. Then the second biggest victory is locked in, say B >
 * D. And then the third, as long as it does not create a voting cycle. In this
 * case, D > C would be ignored because that would say C>B>D>C. The rule
 * considers a big margin of votes (and voters) more certain and forceful than a
 * small margin.</li>
 * <li>Tideman's Condorcet-Hare hybrid elects the Condorcet winner if there is
 * one. If there is a voting-cycle tie, it eliminates all candidates outside the
 * Smith Set. Then it eliminates the candidate who now ranks at the top of the
 * fewest ballots. These two steps repeat until just one candidate remains.</li>
 * <p>
 * Those last three rules are most resistant to manipulation. In the 1980s,
 * Chamberlin et al published research about the ease and frequency of
 * manipulation. In the 2010s, James Green-Armytage has added significantly to
 * this research by including newer voting rules.
 * 
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 6:01:57 PM, Jun 18, 2016
 */

public class Condorcet extends VotingRule {

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] wins = getWinningNum(getPairwisePreferenceMatrix(profile, items));

		List<T> ranking = new ArrayList<>();
		for (int k : MathLib.getRank(wins, false))
			ranking.add(items[k]);
		return ranking;
	}

	/**
	 * A Condorcet winner always wins in all one-against-one / head-to-head
	 * competitions
	 * 
	 * @param profile
	 * @return Condorcet winner in a preference profile if exists
	 */
	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		T winner = getWinner(profile);
		if (winner == null)
			return null;
		else
			return Arrays.asList(winner);
	}

	public <T> T getWinner(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] wins = getWinningNum(getPairwisePreferenceMatrix(profile, items));
		// beat all others
		int argmax = ArrayUtils.indexOf(wins, items.length - 1);
		if (argmax == -1)
			return null;
		else
			return items[argmax];
	}

	/**
	 * A candidate is a Condorcet loser if (s)he always loses in one-on-one
	 * competitions based on majority rule in the preference profile
	 * 
	 * @param profile
	 * @return Condorcet loser in a preference profile if exists
	 */
	public <T> T getLoser(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] wins = getWinningNum(getPairwisePreferenceMatrix(profile, items));
		// beat by all others
		int argmin = ArrayUtils.indexOf(wins, 0);
		if (argmin == -1)
			return null;
		else
			return items[argmin];
	}

	/**
	 * Each element in pairwise preference matrix is a nonnegative number of
	 * tournaments of a candidate out ranks another one during all head-to-head
	 * contests with a specific preference profile.
	 * 
	 * @param profile
	 * @param sortedItems
	 *            eliminate a redundant procedure to sort items for all profiles
	 * @return Pairwise Preference Matrix
	 */
	public static <T> int[][] getPairwisePreferenceMatrix(Profile<T> profile, T[] sortedItems) {
		int numItem = profile.getNumItem();
		if (sortedItems == null)
			sortedItems = profile.getSortedItems();

		int[][] ppm = new int[numItem][numItem];
		for (int k = 0; k < profile.data.length; k++) {
			T[] preference = profile.data[k];
			for (int i = 0; i < numItem; i++) {
				int runner = Arrays.binarySearch(sortedItems, preference[i]);
				for (int j = i + 1; j < numItem; j++) {
					int opponent = Arrays.binarySearch(sortedItems, preference[j]);
					ppm[runner][opponent] += profile.votes[k];
				}
			}
		}
		return ppm;
	}

	/**
	 * Compute the winning number of each item based on its pairwise comparisons
	 * records. TODO: an even number of candidates may lead to tied points.
	 * 
	 * @param ppm
	 *            pairwise preference matrix
	 * @return pairwise winning number
	 */
	public int[] getWinningNum(int[][] ppm) {
		int numItem = ppm.length;
		int[] wins = new int[numItem];

		for (int i = 0; i < numItem; i++)
			for (int j = i + 1; j < numItem; j++) {
				if (ppm[i][j] > ppm[j][i])
					wins[i]++;
				else if (ppm[i][j] < ppm[j][i])
					wins[j]++;
			}
		return wins;
	}
}