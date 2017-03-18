package com.horsehour.vote.rule;

import java.io.Serializable;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.horsehour.vote.Profile;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:40:03 PM, Jun 26, 2016
 */

public class LearnedRule extends VotingRule implements Serializable {
	private static final long serialVersionUID = -7649740770312574806L;
	/**
	 * Abstract election mechanism
	 */
	Function<Profile<?>, List<?>> mechanism;

	/**
	 * Abstract election agent with input profile and alternatives
	 */
	BiFunction<Profile<Integer>, List<Integer>, List<Integer>> agent;

	String name = "";

	@SuppressWarnings("unchecked")
	public <T> LearnedRule(Function<Profile<T>, List<T>> mechanism) {
		this.mechanism = profile -> mechanism.apply((Profile<T>) profile);
	}

	public LearnedRule(BiFunction<Profile<Integer>, List<Integer>, List<Integer>> agent) {
		this.agent = (profile, items) -> agent.apply(profile, items);
	}

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		return getAllWinners(profile);
	}

	@SuppressWarnings("unchecked")
	public <T> List<T> getAllWinners(Profile<T> profile) {
		return (List<T>) mechanism.apply(profile);
	}

	public List<Integer> getAllWinners(Profile<Integer> profile, List<Integer> items) {
		return agent.apply(profile, items);
	}

	public void setName(String name) {
		this.name = name;
	}

	public String toString() {
		if (name.isEmpty())
			return super.toString();
		else
			return name;
	}
}