package com.horsehour.vote.rule;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.collections4.iterators.PermutationIterator;

import com.horsehour.vote.Profile;
import com.horsehour.vote.Sampling;

/**
 * The Kemeny–Young method is a voting system that uses preferential ballots and
 * pairwise comparison counts to identify the most popular choices in an
 * election. It is a Condorcet method because if there is a Condorcet winner, it
 * will always be ranked as the most popular choice.
 * 
 * The Kemeny–Young method was developed by John Kemeny in 1959. Young and
 * Levenglick (1978) showed that this method was the unique neutral method
 * satisfying reinforcement and the Condorcet criterion. In other papers (Young
 * 1986, 1988, 1995, 1997), Young adopted an epistemic approach to
 * preference-aggregation: he supposed that there was an objectively 'correct',
 * but unknown preference order over the alternatives, and voters receive noisy
 * signals of this true preference order (cf. Condorcet's jury theorem). Using a
 * simple probabilistic model for these noisy signals, Young showed that the
 * Kemeny–Young method was the maximum likelihood estimator of the true
 * preference order. Young further argues that Condorcet himself was aware of
 * the Kemeny-Young rule and its maximum-likelihood interpretation, but was
 * unable to clearly express his ideas.
 * 
 * @author Andrew Mao
 * @author Chunheng Jiang
 */
public class KemenyYoung extends VotingRule {

	/**
	 * @param profile
	 * @return all rankings which are nearest to the profile as a whole
	 */
	public <T> List<List<T>> getAllNearestRankings(Profile<T> profile) {
		T[] items = profile.getSortedItems();
		int numItem = profile.getNumItem();
		int[][] ppm = Condorcet.getPairwisePreferenceMatrix(profile, items);

		List<List<Integer>> nearestRankings = new ArrayList<>();
		long nearestDistance = Long.MAX_VALUE;

		List<Integer> list = IntStream.range(0, numItem).boxed().collect(Collectors.toList());
		PermutationIterator<Integer> iter = new PermutationIterator<>(list);
		while (iter.hasNext()) {
			List<Integer> permutate = iter.next();

			long distance = 0;

			for (int i = 0; i < numItem; i++)
				for (int j = i + 1; j < numItem; j++)
					// pairwise preference i > j in the permutation exists
					if (permutate.indexOf(i) < permutate.indexOf(j))
						distance += ppm[j][i];
					else
						distance += ppm[i][j];

			if (distance < nearestDistance) {
				nearestDistance = distance;
				nearestRankings.clear();
			}

			if (distance == nearestDistance)
				nearestRankings.add(permutate);
		}

		// populate the generic rankings with all possible integer-type rankings
		List<List<T>> rankings = new ArrayList<>(nearestRankings.size());
		for (List<Integer> p : nearestRankings) {
			List<T> ranking = new ArrayList<>(numItem);
			for (int i : p)
				ranking.add(items[i]);
			rankings.add(ranking);
		}
		return rankings;
	}

	@Override
	public <T> List<T> getRanking(Profile<T> profile) {
		// select one in random when many exist
		return Sampling.selectRandom(getAllNearestRankings(profile));
	}

	@Override
	public <T> List<T> getAllWinners(Profile<T> profile) {
		List<List<T>> rankings = getAllNearestRankings(profile);
		return rankings.stream().map(ranking -> ranking.get(0)).distinct().collect(Collectors.toList());
	}

	public static void main(String[] args) {
		String[][] preferences = { { "a", "b", "c", "d" }, { "d", "a", "b", "c" }, { "c", "d", "a", "b" },
				{ "b", "c", "d", "a" } };
		int[] votes = { 10, 7, 6, 3 };

		KemenyYoung rule = new KemenyYoung();
		Profile<String> profile = new Profile<>(preferences, votes);

		List<List<String>> result = rule.getAllNearestRankings(profile);
		for(List<String> rt : result)
			System.out.println(rt);
	}
}
