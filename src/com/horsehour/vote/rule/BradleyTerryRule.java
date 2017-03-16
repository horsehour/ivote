package com.horsehour.vote.rule;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;
import com.horsehour.vote.models.BradleyTerryModel;

public class BradleyTerryRule extends ScoredVotingRule {

	BradleyTerryModel bt;
	boolean useAllPairs;

	public BradleyTerryRule(boolean useAllPairs) {
		this.useAllPairs = useAllPairs;
		bt = new BradleyTerryModel();
	}

	@Override
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {
		return bt.fitModel(profile, useAllPairs).getValueMap();
	}

	public String toString() {
		return useAllPairs ? "BTAllP" : "BTAdjP";
	}

}
