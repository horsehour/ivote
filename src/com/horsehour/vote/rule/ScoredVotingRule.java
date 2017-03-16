package com.horsehour.vote.rule;

import java.util.List;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;

public abstract class ScoredVotingRule extends VotingRule {

	/**
	 * @param profile
	 * @return ranking of all candidates in profile with scores
	 */
	public abstract <T> ScoredItems<T> getScoredRanking(Profile<T> profile);

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		return getScoredRanking(profile).getRanking();
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		ScoredItems<T> items = getScoredRanking(profile);
		List<T> ranking = items.getRanking();
		double largestScore = items.get(ranking.get(0));
		int i = 1;
		for (; i < profile.getNumItem(); i++)// possible cyclic preferences
			if (items.get(ranking.get(i)) < largestScore)
				break;
		return ranking.subList(0, i);
	}
}