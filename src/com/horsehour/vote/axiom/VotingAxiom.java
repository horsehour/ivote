package com.horsehour.vote.axiom;

import java.util.List;
import java.util.stream.Stream;

import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.rule.VotingRule;

/**
 * Voting axiomatic property is used to evaluate the fairness of a voting rule
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 10:40:01 AM, Jun 17, 2016
 *
 */

public abstract class VotingAxiom {
	/**
	 * Compute the satisfiability of one voting rule to the criterion
	 * 
	 * @param numItem
	 * @param numVote
	 * @param rule
	 * @return satisfiability of the voting rule to the criterion
	 */
	public double getSatisfiability(int numItem, int numVote, VotingRule rule) {
		Stream<Profile<Integer>> profiles = DataEngine.getAECPreferenceProfiles(numItem, numVote);
		return getSatisfiability(profiles, rule);
	}

	/**
	 * Compute the satisfiability of one voting rule to the criterion based on
	 * random samples
	 * 
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param rule
	 * @return satisfiability of the voting rule to the criterion
	 */
	public double getSatisfiability(int numItem, int numVote, int numSample, VotingRule rule) {
		List<Profile<Integer>> profiles = DataEngine.getRandomProfiles(numItem, numVote, numSample);
		return getSatisfiability(profiles.stream(), rule);
	}

	/**
	 * Compute the satisfiability of one voting rule to the criterion based upon
	 * random samplings
	 * 
	 * @param profiles
	 * @param rule
	 * @return satisfiability of the voting rule to the criterion
	 */
	public double getSatisfiability(List<Profile<Integer>> profiles, VotingRule rule) {
		return getSatisfiability(profiles.stream(), rule);
	}

	public abstract double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule);

	/**
	 * Evaluate whether the voting rule violates the criterion
	 * 
	 * @param numItem
	 * @param numVote
	 * @param rule
	 * @return true if rule violates the criterion, false otherwise
	 */
	public boolean isViolated(int numItem, int numVote, VotingRule rule) {
		Stream<Profile<Integer>> profiles = DataEngine.getAECPreferenceProfiles(numItem, numVote);
		return isViolated(profiles, rule);
	}

	/**
	 * Evaluate whether the voting rule violates the criterion based on random
	 * samples
	 * 
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param rule
	 * @return true if rule violates the criterion, false otherwise
	 */
	public boolean isViolated(int numItem, int numVote, int numSample, VotingRule rule) {
		List<Profile<Integer>> profiles = DataEngine.getRandomProfiles(numItem, numVote, numSample);
		return isViolated(profiles, rule);
	}

	/**
	 * Evaluate whether the voting rule violates the criterion
	 * 
	 * @param profiles
	 * @param rule
	 * @return true if rule violates the criterion, false otherwise
	 */
	public boolean isViolated(List<Profile<Integer>> profiles, VotingRule rule) {
		return isViolated(profiles.stream(), rule);
	}

	public abstract boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule);

	public String toString() {
		return getClass().getSimpleName();
	}
}
