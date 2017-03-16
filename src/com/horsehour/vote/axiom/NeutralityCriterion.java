package com.horsehour.vote.axiom;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.VotingRule;

/**
 * To evaluate neutrality criterion, we should impose a permutation over
 * candidates and alter all voters' preferences of candidates based on the
 * permutation. If a voting rule violates the rule, that is to say when it is
 * applied to the permutated profile, it doesn't select the corresponding winner
 * according to the permutation rule. For example, if all voters switch their
 * preferences over a and b, that will swap the places of a and b in all
 * preferences. If the voting rule select a as the winner before the
 * permutation. After the permutation, if the voting rule as a social choice
 * function could could always select b as the winner, or as a social welfare
 * function, could produce a consistent permuated social preference over
 * candidates, we therefore safely arrive to the conclusion that it satisfies
 * the neutrality criterion.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 10:59:04 AM, Jun 17, 2016
 */

public class NeutralityCriterion extends VotingAxiom {
	/**
	 * Predicted winners should be the same as the permuted winners
	 */
	BiPredicate<List<?>, List<?>> violator = (permutedWinners, predicted) -> {
		if (predicted == null)
			return true;
		else if (predicted.size() <= permutedWinners.size())
			return predicted.stream().anyMatch(c -> !permutedWinners.contains(c));
		else
			return permutedWinners.stream().anyMatch(c -> !predicted.contains(c));
	};

	int numItem;

	@Override
	public double getSatisfiability(int numItem, int numVote, VotingRule rule) {
		this.numItem = numItem;
		return super.getSatisfiability(numItem, numVote, rule);
	}

	public double getSatisfiability(List<Profile<Integer>> profiles, VotingRule rule) {
		this.numItem = profiles.get(0).getNumItem();
		return super.getSatisfiability(profiles, rule);
	}

	public double getSatisfiability(Stream<Profile<Integer>> profiles, VotingRule rule) {
		List<List<Integer>> permutations = DataEngine.getAllPermutations(numItem);

		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			// long stat = profile.getStat();
			long match = permuteAndEvalMatch(profile, permutations, rule);
			if (match == -1)// no winner at all
				return;

			numTotal.addAndGet(permutations.size());
			numMatch.addAndGet(match);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	<T> long permuteAndEvalMatch(Profile<T> profile, List<List<T>> permutations, VotingRule rule) {
		List<T> winners = rule.getAllWinners(profile);
		if (winners == null)
			return -1;

		int len = profile.data.length;
		T[] pref = profile.data[0];

		long count = 0;
		for (int k = 0; k < permutations.size(); k++) {
			Map<T, T> map = getMap(pref, permutations.get(k));
			if (map == null) {// identity function
				count++;
				continue;
			}

			T[][] preferences = Arrays.copyOf(profile.data, len);
			for (int i = 0; i < len; i++)
				preferences[i] = applyMap(profile.data[i], map);

			Profile<T> permutedProfile = new Profile<>(preferences, profile.votes);
			List<T> predicted = rule.getAllWinners(permutedProfile);
			List<T> permutedWinners = applyMap(winners, map);
			if (!violator.test(permutedWinners, predicted))
				count++;
		}
		return count;
	}

	public <T> List<Pair<Profile<T>, List<T>>> getAllPermutedProfiles(Profile<T> profile, List<T> winners,
			List<List<T>> permutations) {

		int len = profile.data.length;
		T[] preference = profile.data[0];

		List<Pair<Profile<T>, List<T>>> ret = new ArrayList<>();

		for (List<T> permutation : permutations) {
			Map<T, T> map = getMap(preference, permutation);
			// identity map
			if (map == null) {
				ret.add(Pair.of(profile, winners));
				continue;
			}

			T[][] preferences = Arrays.copyOf(profile.data, profile.data.length);
			for (int i = 0; i < len; i++)
				preferences[i] = applyMap(profile.data[i], map);

			Profile<T> permutedProfile = new Profile<>(preferences, profile.votes);
			// permuted profile and permuted winners
			ret.add(Pair.of(permutedProfile, applyMap(winners, map)));
		}
		return ret;
	}

	<T> Map<T, T> getMap(T[] preference, List<T> permutation) {
		Map<T, T> map = new HashMap<>();
		int count = 0;
		for (int i = 0; i < preference.length; i++) {
			map.put(preference[i], permutation.get(i));
			if (preference[i] == permutation.get(i))
				count++;
		}
		if (count == preference.length)// identity map
			return null;
		return map;
	}

	<T> T[] applyMap(T[] preference, Map<T, T> map) {
		T[] result = Arrays.copyOf(preference, preference.length);
		for (int i = 0; i < preference.length; i++)
			result[i] = map.get(preference[i]);
		return result;
	}

	<T> List<T> applyMap(List<T> items, Map<T, T> map) {
		List<T> result = new ArrayList<>();
		for (int i = 0; i < items.size(); i++)
			result.add(map.get(items.get(i)));
		return result;
	}

	@Override
	public boolean isViolated(int numItem, int numVote, VotingRule rule) {
		this.numItem = numItem;
		return super.isViolated(numItem, numVote, rule);
	}

	@Override
	public boolean isViolated(List<Profile<Integer>> profiles, VotingRule rule) {
		this.numItem = profiles.get(0).getNumItem();
		return super.isViolated(profiles, rule);
	}

	public boolean isViolated(Stream<Profile<Integer>> profiles, VotingRule rule) {
		List<List<Integer>> permutations = DataEngine.getAllPermutations(numItem);
		return profiles.parallel().anyMatch(profile -> permuteAndEvalViolation(profile, permutations, rule));
	}

	<T> boolean permuteAndEvalViolation(Profile<T> profile, List<List<T>> permutations, VotingRule rule) {
		List<T> winners = rule.getAllWinners(profile);
		if (winners == null)
			return false;

		int len = profile.data.length;
		T[] pref = profile.data[0];

		for (int k = 0; k < permutations.size(); k++) {
			Map<T, T> map = getMap(pref, permutations.get(k));
			if (map == null)// identity map
				continue;

			T[][] preferences = Arrays.copyOf(profile.data, len);
			for (int i = 0; i < len; i++)
				preferences[i] = applyMap(profile.data[i], map);

			Profile<T> permutedProfile = new Profile<>(preferences, profile.votes);
			List<T> predicted = rule.getAllWinners(permutedProfile);
			List<T> permutedWinners = applyMap(winners, map);
			if (violator.test(permutedWinners, predicted))
				return true;
		}
		return false;
	}
}