package com.horsehour.vote.rule;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;

/**
 * @author Andrew Mao
 */
public abstract class PositionalVotingRule extends ScoredVotingRule {
	/**
	 * @param length
	 * @return position related scores over a ranking in certain length 
	 */
	protected abstract double[] getPositionalScores(int length);

	@Override
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {
		double[] scores = getPositionalScores(profile.getNumItem());

		ScoredItems<T> items = new ScoredItems<>(profile.getSortedItems());

		int id = 0;
		for (T[] pref : profile.data) {
			for (int i = 0; i < pref.length; i++)
				items.put(pref[i], items.get(pref[i]) + scores[i] * profile.votes[id]);
			id++;
		}
		return items;
	}
}
