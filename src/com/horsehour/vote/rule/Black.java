package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.vote.Profile;

/**
 * Duncan Black (23 May 1908 – 14 January 1991) was a Scottish economist who
 * laid the foundations of social choice theory. In particular he was
 * responsible for unearthing the work of many early political scientists,
 * including Charles Lutwidge Dodgson, and was responsible for the Black
 * electoral system, a Condorcet method whereby, in the absence of a Condorcet
 * winner (e.g. due to a cycle), the Borda winner is chosen.
 * <p>
 * We could define other “hybrid” methods, where we choose the Condorecet winner
 * if one exists and use some other method if there is no Condorcet winner.
 * Perhaps a hybrid method is the way to go in order to ensure that reasonable
 * properties hold for our chosen method.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 12:47:47 PM, Aug 1, 2016
 *
 */

public class Black extends Condorcet {
	Condorcet condorcet = new Condorcet();
	Borda borda = new Borda();

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int[] wins = getWinningNum(getPairwisePreferenceMatrix(profile, items));

		List<T> ranking = new ArrayList<>();
		for (int k : MathLib.getRank(wins, false))
			ranking.add(items[k]);
		return ranking;
	}

	public <T> List<T> getAllWinners(Profile<T> profile) {
		List<T> winners = condorcet.getAllWinners(profile);
		if (winners == null)
			winners = borda.getAllWinners(profile);
		return winners;
	}
}
