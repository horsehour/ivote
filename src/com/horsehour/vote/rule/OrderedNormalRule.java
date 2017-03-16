package com.horsehour.vote.rule;

import org.apache.commons.math3.distribution.NormalDistribution;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;
import com.horsehour.vote.models.OrderedNormalMCEM;

public class OrderedNormalRule<T> extends ScoredVotingRule {

	static final int MAX_ITERS = 30;
	static final double ABS_EPS = 1e-4; // Only want order, don't care too much
										// about actual scores
	static final double REL_EPS = 1e-5;

	@SuppressWarnings("hiding")
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {
		OrderedNormalMCEM model = new OrderedNormalMCEM(true, MAX_ITERS, ABS_EPS, REL_EPS);

		model.setup(new NormalDistribution(0, 1).sample(4));

		return model.fitModelOrdinal(profile).getValueMap();
	}
}
