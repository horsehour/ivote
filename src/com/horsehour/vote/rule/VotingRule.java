package com.horsehour.vote.rule;

import java.util.List;

import com.horsehour.vote.Profile;

/**
 * A voting rule has two optional outputs: a social choice and a social
 * preference, social choice is generated based on a social choice function or
 * mechanism based on all voters' preference profiles over a set of items,
 * however the later one represents an overall preference plays as a compromised
 * result among all voters' opinions.
 * 
 * @author Chunheng Jiang
 */
public abstract class VotingRule {
	/**
	 * create a ranking over all candidates in the profile
	 * 
	 * @param profile
	 * @return ranking of all candidates in the profile
	 */
	public abstract <T> List<T> getRanking(Profile<T> profile);

	/**
	 * @param profile
	 * @return all winners allowed in the profile
	 */
	public abstract <T> List<T> getAllWinners(Profile<T> profile);

	/**
	 * @return name of the voting rule
	 */
	public String toString() {
		return getClass().getSimpleName();
	}
}
