package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

import org.apache.commons.collections4.iterators.PermutationIterator;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;

/**
 * Elect all possible STV winners using brute force approach. The ultimate goal
 * is to set m!-priority rankings over m candidates, and each priority ranking
 * is considered as a possible tie-breaking rule. Given each possible
 * tie-breaking rule, STV may select a possible different winner. We have all
 * the possible winners with all m!-tie-breaking rules, and using the
 * frequencies that a candidate being selected as the winner to indicate the
 * probability of the candidates being a winner in a random picking.
 * <p>
 * The greatest disadvantage is that when the number of candidates m becomes
 * larger, it will extremely hard to computed the winners.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 9:08:08 PM, Jan 6, 2017
 */
public class BruteForceSTV {
	public long begin;
	public float elapsed;

	Profile<Integer> profile;
	List<Integer> permutation;

	public int numItem;
	public List<Integer> candidates;
	public List<Integer> elected;
	public Map<Integer, Integer> countElected;

	public BruteForceSTV() {}

	/**
	 * Choose a remaining tied candidate based on predefined ranking priority,
	 * i.e. tie-breaking rule
	 * 
	 * @param tiedRemaining
	 * @return eliminated candidate at next round
	 */
	int getEliminated(List<Integer> tiedRemaining) {
		int min = numItem;
		for (int tied : tiedRemaining) {
			int index = permutation.indexOf(tied);
			if (index < min)
				min = index;
		}
		return permutation.get(min);
	}

	int[] scoring(Profile<Integer> profile, List<Integer> candidates) {
		int[] votes = profile.votes;
		int[] scores = new int[candidates.size()];

		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = 0; i < pref.length; i++) {
				int item = pref[i];
				int index = candidates.indexOf(item);
				// item has been eliminated
				if (index == -1)
					continue;
				scores[index] += votes[c];
				break;
			}
			c++;
		}
		return scores;
	}

	void elect(List<Integer> remaining) {
		if (remaining.size() == 1) {
			int item = remaining.get(0);
			elected.add(item);
			return;
		}

		int[] scores = scoring(profile, remaining);
		TreeMap<Integer, List<Integer>> tiers = new TreeMap<>(Collections.reverseOrder());
		for (int i = 0; i < scores.length; i++) {
			int score = scores[i];
			List<Integer> member = null;
			if (tiers.containsKey(score))
				member = tiers.get(score);
			else
				member = new ArrayList<>();
			member.add(remaining.get(i));
			tiers.put(score, member);
		}

		int max = tiers.firstKey();
		if (max > profile.numVoteTotal / 2) {
			int item = tiers.get(max).get(0);
			elected.add(item);
			return;
		}

		int min = tiers.lastKey();
		List<Integer> tied = tiers.get(min);
		Integer eliminated = getEliminated(tied);
		remaining.remove(eliminated);
		elect(remaining);
	}

	/**
	 * @param profile
	 * @return All possible winners with various tie-breaking rules
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile) {
		this.profile = profile;

		numItem = profile.getNumItem();

		candidates = Arrays.asList(profile.getSortedItems());

		elected = new ArrayList<>();

		PermutationIterator<Integer> iter = new PermutationIterator<>(candidates);

		begin = System.currentTimeMillis();

		while (iter.hasNext()) {
			permutation = iter.next();
			elect(new ArrayList<>(candidates));
		}
		elapsed = System.currentTimeMillis() - begin;

		return elected.stream().distinct().sorted().collect(Collectors.toList());
	}

	/**
	 * @param profile
	 * @param numRTB
	 *            number of random tie-breaking rules draw from m! permutations
	 * @return all possible winners based on these tie-breaking rules
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile, int numRTB) {
		if (numRTB <= 0)
			return getAllWinners(profile);

		this.profile = profile;
		numItem = profile.getNumItem();
		candidates = Arrays.asList(profile.getSortedItems());

		elected = new ArrayList<>();

		begin = System.currentTimeMillis();

		for (int i = 0; i < numRTB; i++) {
			permutation = Arrays.asList(DataEngine.getRandomPermutation(numItem));
			elect(new ArrayList<>(candidates));
		}
		elapsed = (System.currentTimeMillis() - begin) / 1000.0F;

		countElected = new HashMap<>();
		for (int item : candidates)
			countElected.put(item, 0);

		for (int item : elected)
			countElected.put(item, countElected.get(item) + 1);

		List<Integer> winners = new ArrayList<>();
		for (int item : candidates) {
			if (countElected.get(item) == 0)
				continue;
			else
				winners.add(item);
		}
		return winners;
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/documents/csc/", dataset = "soc-3", file = "M10N10-3.csv";
		Profile<Integer> profile = DataEngine.loadProfile(Paths.get(base + dataset + "/" + file));

		BruteForceSTV rule = new BruteForceSTV();
		System.out.println(rule.getAllWinners(profile, 10));

		TickClock.stopTick();
	}
}