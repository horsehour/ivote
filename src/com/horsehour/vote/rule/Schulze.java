package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;

/**
 * The Schulze method is a voting system developed in 1997 by Markus Schulze
 * that selects a single winner using votes that express preferences. The method
 * can also be used to create a sorted list of winners. The Schulze method is
 * also known as Schwartz Sequential Dropping (SSD), Cloneproof Schwartz
 * Sequential Dropping (CSSD), the Beatpath Method, Beatpath Winner, Path
 * Voting, and Path Winner.
 * <p>
 * The Schulze method is a Condorcet method, which means the following: if there
 * is a candidate who is preferred by a majority over every other candidate in
 * pairwise comparisons, then this candidate will be the winner when the Schulze
 * method is applied.
 * <p>
 * The output of the Schulze method (defined below) gives an ordering of
 * candidates. Therefore, if several positions are available, the method can be
 * used for this purpose without modification, by letting the k top-ranked
 * candidates win the k available seats. Furthermore, for proportional
 * representation elections, a single transferable vote variant has been
 * proposed.
 * <p>
 * This is a method that evolved in a sequence of email and Internet postings by
 * a group of enthusiasts who sought to develop workable voting methods that can
 * actually be put to use in “real life.” Schulze notes that this method is used
 * in organizations that have an aggregate membership of more than 1,700 and
 * that it is the most widely-used of all of the Condorcet round-robin methods.
 * People who use the Linux operating system might be interested to know that
 * the Schulze method is used in decision making in the Debian Linux project,
 * one of the most widely disseminated Linux distributions.
 * <p>
 * The Schulze method is used by several organizations including Debian, Ubuntu,
 * Gentoo, Software in the Public Interest, Free Software Foundation Europe,
 * Pirate Party political parties and many others.
 * <p>
 * The Schulze method violates the consistency criterion, the participation
 * criterion.
 * 
 * @see
 *      <li>'A new monotonic, clone-independent, reversal symmetric, and
 *      Condorcet-consistent single-winner election method ', Markus Schulze,
 *      2011.</li>
 *      <li>'Schulze and Ranked-Pairs Voting are Fixed-Parameter Tractable to
 *      Bribe, Manipulate, and Control', Lane A. Hemaspaandra, Rahman Lavaee and
 *      Curtis Menton, 2012.</li>
 *      <li>'Manipulation and Control Complexity of Schulze Voting', Curtis
 *      Menton and Preetjot Singh, 2012.</li>
 *      <li>'A Complexity-of-Strategic-Behavior Comparison between Schulze’s
 *      Rule and Ranked Pairs', David Parkes and Lirong Xia, 2012.</li>
 *      <li>'Coalitional Manipulation for Schulze’s Rule', Serge Gaspers, Thomas
 *      Kalinowski, Nina Narodytska and Toby Walsh, 2013.</li>
 * @author Chunheng Jiang
 * @version 1.0
 * @since 12:49:51 PM, Aug 1, 2016
 *
 */

public class Schulze extends Condorcet {
	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[][] ppm = getPairwisePreferenceMatrix(profile, items);
		int[][] strengths = getStrongestPathStrength(ppm);

		int m = items.length;
		int[] wins = new int[m];
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < m; j++) {
				if (strengths[i][j] >= strengths[j][i])
					wins[i]++;
				else
					wins[j]++;
			}
		}

		int[] rank = MathLib.getRank(wins, false);
		List<T> prefList = new ArrayList<>();
		for (int r : rank)
			prefList.add(items[r]);
		return prefList;
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[][] ppm = getPairwisePreferenceMatrix(profile, items);
		int[][] strengths = getStrongestPathStrength(ppm);

		int m = items.length;
		int[] wins = new int[m];
		// Original Schulze method breaks ties in random
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < m; j++) {
				if (strengths[i][j] > strengths[j][i])
					wins[i]++;
				else if (strengths[i][j] < strengths[j][i])
					wins[j]++;
			}
		}

		int[] argmax = MathLib.argmax(wins);
		List<T> winners = new ArrayList<>();
		for (int i : argmax)
			winners.add(items[i]);
		return winners;
	}

	/**
	 * Computing the strongest path strengths. It is a well-known problem in
	 * graph theory, sometimes called the Widest Path problem. One simple way to
	 * compute the strengths therefore is a variant of the Floyd–Warshall
	 * algorithm.
	 * 
	 * @param ppm
	 *            pairwise preference matrix
	 * @return strongest path strengths for each pair of candidates
	 */
	int[][] getStrongestPathStrength(int[][] ppm) {
		int m = ppm.length;
		int[][] strengths = new int[m][m];
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < m; j++) {
				if (ppm[i][j] > ppm[j][i])
					strengths[i][j] = ppm[i][j];
				else if (ppm[i][j] < ppm[j][i])
					strengths[j][i] = ppm[j][i];
			}
		}

		/**
		 * a chain is only as strong as its weakest link
		 */
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				if (i == j)
					continue;
				for (int k = 0; k < m; k++) {
					if (i == k || j == k)
						continue;
					strengths[j][k] = Math.max(strengths[j][k], Math.min(strengths[j][i], strengths[i][k]));
				}
			}
		}
		return strengths;
	}
}
