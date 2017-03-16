package com.horsehour.vote.rule;

import java.io.Serializable;
import java.util.List;
import java.util.function.Function;

import com.horsehour.vote.Profile;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:40:03 PM, Jun 26, 2016
 *
 */

public class LearnedRule extends VotingRule implements Serializable {
	private static final long serialVersionUID = -7649740770312574806L;
	Function<Profile<?>, List<?>> mechanism;

	String name = "";

	@SuppressWarnings("unchecked")
	public <T> LearnedRule(Function<Profile<T>, List<T>> mechanism) {
		this.mechanism = profile -> mechanism.apply((Profile<T>) profile);
	}

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		return getAllWinners(profile);
	}

	@SuppressWarnings("unchecked")
	public <T> List<T> getAllWinners(Profile<T> profile) {
		return (List<T>) mechanism.apply(profile);
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