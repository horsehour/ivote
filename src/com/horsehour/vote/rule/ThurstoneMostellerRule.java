package com.horsehour.vote.rule;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;
import com.horsehour.vote.models.ThurstoneMostellerModel;

public class ThurstoneMostellerRule extends ScoredVotingRule {
	ThurstoneMostellerModel tm;
	boolean allPairsUsed;
	
	public ThurstoneMostellerRule(boolean useAllPairs) {
		this.allPairsUsed = useAllPairs;
		tm = new ThurstoneMostellerModel();
	}
	
	public <T> ScoredItems<T> getScoredRanking(Profile<T> profile) {				
		return tm.fitModel(profile, allPairsUsed).getValueMap();								
	}	

	public String toString() { 
		return super.toString() + (allPairsUsed ? "-AllPair" : "-AdjPair");
	}
}