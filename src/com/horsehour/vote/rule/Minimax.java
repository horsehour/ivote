package com.horsehour.vote.rule;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;
import com.horsehour.vote.axiom.MajorityLoserCriterion;
import com.horsehour.vote.axiom.VotingAxiom;

/**
 *
 * Minimax is often considered to be the simplest of the Condorcet methods. It
 * is also known as the Simpson-Kramer method, and the successive reversal
 * method.
 * <p>
 * Minimax selects the candidate for whom the greatest pairwise score for
 * another candidate against him is the least such score among all candidates.
 * Formally, let score(X,Y) denote the pairwise score for X against Y. Then the
 * candidate, W selected by minimax (aka the winner) is given by:<br>
 * W = argmin_X(max_Y score(Y,X)) = argmax_Y(min_X score(Y, X))
 * <p>
 * When it is permitted to rank candidates equally, or to not rank all the
 * candidates, three interpretations of the rule are possible. When voters must
 * rank all the candidates, all three rules are equivalent. The score for
 * candidate x against y can be defined as:
 * <li>The number of voters ranking x above y, but only when this score exceeds
 * the number of voters ranking y above x. If not, then the score for x against
 * y is zero. This is sometimes called winning votes.</li>
 * <li>The number of voters ranking x above y minus the number of voters ranking
 * y above x. This is called using margins.</li>
 * <li>The number of voters ranking x above y, regardless of whether more voters
 * rank x above y or vice versa. This interpretation is sometimes called
 * pairwise opposition.</li>
 * <p>
 * When one of the first two interpretations is used, the method can be restated
 * as: "Disregard the weakest pairwise defeat until one candidate is unbeaten.
 * " An "unbeaten" candidate possesses a maximum score against him which is zero
 * or negative.
 * <p>
 * Minimax using winning votes or margins satisfies Condorcet and the majority
 * criterion, but not the Smith criterion, mutual majority criterion,
 * independence of clones criterion, or Condorcet loser criterion. When winning
 * votes is used, Minimax also satisfies the Plurality criterion. When the
 * pairwise opposition interpretation is used, minimax also does not satisfy the
 * Condorcet criterion. However, when equal-ranking is permitted, there is never
 * an incentive to put one's first-choice candidate below another one on one's
 * ranking. It also satisfies the Later-no-harm criterion, which means that by
 * listing additional, lower preferences in one's ranking, one cannot cause a
 * preferred candidate to lose.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:18:41 PM, Jul 31, 2016
 *
 */

public class Minimax extends ScoredVotingRule {
	/**
	 * mode in fault
	 */
	public Mode mode = Mode.PairwiseOpposition;

	public enum Mode {
		WinningVote, Margin, PairwiseOpposition
	}

	public Minimax() {
	}

	public Minimax(Mode mode) {
		this.mode = mode;
	}

	@Override
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[][] ppm = Condorcet.getPairwisePreferenceMatrix(profile, items);
		if (mode == Mode.WinningVote)
			ppm = getWinningVoteScores(ppm);
		else if (mode == Mode.Margin)
			ppm = getMarginScores(ppm);

		int[] scores = getScores(ppm);
		return new ScoredItems<>(items, scores);
	}

	int[][] getWinningVoteScores(int[][] ppm) {
		int m = ppm.length;
		int[][] scores = new int[m][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				if (i == j)
					continue;
				if (ppm[i][j] > ppm[j][i])
					scores[i][j] = ppm[i][j];
			}
		}
		return scores;
	}

	int[][] getMarginScores(int[][] ppm) {
		int m = ppm.length;
		int[][] scores = new int[m][m];
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < m; j++) {
				scores[i][j] = ppm[i][j] - ppm[j][i];
				scores[j][i] = -scores[i][j];
			}
		}
		return scores;
	}

	int[] getScores(int[][] pairwiseScores) {
		int m = pairwiseScores.length;
		int[] scores = new int[m];

		for (int i = 0; i < m; i++) {
			scores[i] = Integer.MIN_VALUE;
			for (int j = 0; j < m; j++) {
				if (i == j)
					continue;
				
				if (pairwiseScores[j][i] > scores[i])
					scores[i] = pairwiseScores[j][i];
			}
			// for the convenience of sorting in descending order
			scores[i] *= -1;
		}
		return scores;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		int numItem = 5, numVote = 107, numSample = 5000;

		VotingRule rule = new Minimax();
		VotingAxiom axiom = new MajorityLoserCriterion();

		System.out.println(axiom.getSatisfiability(numItem, numVote, numSample, rule));

		TickClock.stopTick();
	}
}