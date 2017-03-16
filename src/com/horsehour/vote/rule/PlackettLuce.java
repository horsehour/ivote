package com.horsehour.vote.rule;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;
import com.horsehour.vote.models.PlackettLuceModel;

public class PlackettLuce extends ScoredVotingRule {

	double LARGE_SPACER = 1e6;

	@Override
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile){
		// Just fit approximate parameters in case of non-convergence
		PlackettLuceModel model = new PlackettLuceModel(false);

		return model.fitModelOrdinal(profile).getValueMap();
	}
}
