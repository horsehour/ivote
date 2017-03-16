package com.horsehour.vote;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.CombinatoricsUtils;

import com.horsehour.util.TickClock;

/**
 * Weak preference profile over candidates/items by voter. The most important in
 * the structure is all preference ranking of voters. Since many people may
 * share with others the same preference ranking, therefore, we introduce the
 * votes for each preference ranking to reduce the computing time.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Dec. 20, 2016
 */
public class ProfileWeak {
	/**
	 * preference rankings the profile contains
	 */
	public Candidate[][] data;
	/**
	 * distribution of votes of corresponding preference rankings, default 1
	 */
	public int[] votes;
	public int numVote;

	public ProfileWeak(Candidate[][] dat) {
		this.data = dat;
		numVote = dat.length;
		votes = new int[numVote];
		Arrays.fill(votes, 1);
	}

	public ProfileWeak(Candidate[][] dat, int[] votes) {
		this.data = dat;
		this.votes = votes;

		int sum = 0;
		for (int k : votes)
			sum += k;
		numVote = sum;
	}

	public ProfileWeak(List<List<Candidate>> dat, int[] votes) {
		int numRanking = dat.size();
		int numItem = dat.get(0).size();

		this.data = new Candidate[numRanking][numItem];
		this.votes = votes;
		int sum = 0;
		for (int i = 0; i < numRanking; i++) {
			for (int j = 0; j < numItem; j++)
				data[i][j] = dat.get(i).get(j);
			sum += votes[i];
		}
		this.numVote = sum;
	}

	/**
	 * @return all preferences' hash codes
	 */
	public int[] getFingerPrint() {
		int[] hashCodes = new int[votes.length];
		for (int i = 0; i < votes.length; i++)
			hashCodes[i] = Arrays.hashCode(data[i]);
		return hashCodes;
	}

	public ProfileWeak merge(ProfileWeak profile) {
		int[] indices = new int[profile.data.length];
		int i = 0, count = 0;

		int[] fingerprint = getFingerPrint();
		for (int hashcode : profile.getFingerPrint()) {
			indices[i] = ArrayUtils.indexOf(fingerprint, hashcode);
			if (indices[i] == -1)
				count++;
			i++;
		}

		int numItem = profile.getNumItem();
		int numRanking = data.length + count;
		Candidate[][] preferences = new Candidate[numRanking][numItem];
		int[] voteList = new int[numRanking];

		for (int k = 0; k < data.length; k++) {
			voteList[k] = votes[k];
			preferences[k] = Arrays.copyOf(data[k], numItem);
		}

		i = data.length;
		for (int k = 0; k < indices.length; k++) {
			int index = indices[k];
			if (index == -1) {
				preferences[i] = Arrays.copyOf(profile.data[k], numItem);
				voteList[i] = profile.votes[k];
				i++;
			} else
				voteList[index] += profile.votes[k];
		}
		return new ProfileWeak(preferences, voteList);
	}

	public Candidate[][] getData() {
		return data;
	}

	/**
	 * number of profiles with the same pattern in the full space
	 * 
	 * @return
	 */
	public long getStat() {
		int sum = 0;
		long prod = 1;
		for (int vote : votes) {
			int remain = numVote - sum;
			sum += vote;
			prod *= CombinatoricsUtils.binomialCoefficient(remain, vote);
		}
		return prod;
	}

	public float getStatProb() {
		int sum = 0;
		long factorial = CombinatoricsUtils.factorial(getNumItem());
		float prod = 1;
		for (int vote : votes) {
			int remain = numVote - sum;
			sum += vote;
			prod *= CombinatoricsUtils.binomialCoefficient(remain, vote) / Math.pow(factorial, vote);
		}
		return prod;
	}

	public List<Integer> getSortedItems() {
		List<Integer> items = new ArrayList<>();
		for (Candidate[] candidate : data) {
			for (int i = 0; i < candidate.length; i++) {
				int id = candidate[i].id;
				items.add(id);
			}
		}

		items = items.stream().distinct().collect(Collectors.toList());
		Collections.sort(items);
		return items;
	}

	public int getNumItem() {
		return data[0].length;
	}

	public int getNumVote() {
		return numVote;
	}

	/**
	 * Get the indices of preferences with respect to some ordering
	 * 
	 * @param ordering
	 * @return
	 */
	public List<int[]> getIndices(List<Candidate> ordering) {
		List<int[]> rankings = new ArrayList<>();

		for (Candidate[] preference : data) {
			int[] ranking = new int[preference.length];
			int i = 0;
			for (Candidate item : preference)
				ranking[i++] = ordering.indexOf(item) + 1;
			rankings.add(ranking);
		}

		return rankings;
	}

	/**
	 * Returns the number of times each candidate has appeared in a particular
	 * place, with an array in the same order as the initial data.
	 * 
	 * @return
	 */
	public Map<Candidate, int[]> getPositionCounts() {
		Map<Candidate, int[]> counts = new TreeMap<>();

		for (Candidate t : data[0])
			counts.put(t, new int[data[0].length]);

		for (Candidate[] ranking : data) {
			for (int i = 0; i < ranking.length; i++)
				counts.get(ranking[i])[i]++;
		}

		return counts;
	}

	/**
	 * Check how many of the actual pairwise rankings agree with the comparator.
	 * 
	 * @param first
	 * @param second
	 * @param comp
	 * @return
	 */
	public int getNumCorrect(Candidate first, Candidate second, Comparator<Candidate> comp) {
		int correct = 0;

		for (Candidate[] ranking : data) {
			int idxFirst = ArrayUtils.indexOf(ranking, first);
			int idxSecond = ArrayUtils.indexOf(ranking, second);

			if (idxFirst == ArrayUtils.INDEX_NOT_FOUND || idxSecond == ArrayUtils.INDEX_NOT_FOUND)
				continue;

			int c = comp.compare(first, second);

			if (c < 0 && idxFirst < idxSecond)
				correct++;
			else if (c > 0 && idxFirst > idxSecond)
				correct++;
			else if (c == 0 && idxFirst == idxSecond)
				correct++;
		}

		return correct;
	}

	/**
	 * reduces the preference profile to a smaller subsample
	 * 
	 * @param subsetSize
	 */
	public ProfileWeak copyRandomSubset(int subsetSize, Random rnd) {
		if (subsetSize < data.length)
			return new ProfileWeak(Sampling.selectKRandom(data, subsetSize, rnd));
		else
			return this;
	}

	public ProfileWeak slice(int fromIdx, int toIdx) {
		int newSize = toIdx - fromIdx;
		Candidate[][] newProfile = new Candidate[data.length][newSize];
		for (int i = 0; i < data.length; i++)
			System.arraycopy(data[i], fromIdx, newProfile[i], 0, newSize);
		return new ProfileWeak(newProfile);
	}

	public static ProfileWeak singleton(List<Candidate> ranking) {
		Candidate[][] arr2 = new Candidate[1][];
		ranking.toArray(arr2[0]);
		return new ProfileWeak(arr2);
	}

	public boolean equals(ProfileWeak profile) {
		return hashCode() == profile.hashCode();
	}

	public int hashCode() {
		Map<Integer, Integer> cluster = IntStream.range(0, votes.length).boxed()
				.map(i -> Pair.of(Arrays.hashCode(data[i]), votes[i]))// map
				.sorted((p1, p2) -> p1.getLeft().compareTo(p2.getLeft()))// sort
				.collect(Collectors.groupingBy(p -> p.getLeft(), Collectors.summingInt(p -> p.getRight())));

		HashCodeBuilder hcb = new HashCodeBuilder(17, 37);
		for (int key : cluster.keySet())
			hcb.append(new int[] { key, cluster.get(key) });
		return hcb.hashCode();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < data.length; i++) {
			sb.append(votes[i]).append(":");
			sb.append(Arrays.toString(data[i]));
			if (i == data.length - 1)
				break;
			sb.append("\t");
		}
		return sb.toString();
	}

	public static void main(String[] args) {
		TickClock.beginTick();
		// int numFactorial = 6, numVote = 5, numSample = 10000;
		// double[] prob = new double[numFactorial];
		// Arrays.fill(prob, 1.0 / numFactorial);
		//
		// int[][] samples = MathUtils.getMultinomialSamples(prob, numVote,
		// numSample);
		//
		// Map<String, Long> countTable =
		// Arrays.stream(samples).map(Arrays::toString)
		// .collect(Collectors.groupingBy(e -> e, Collectors.counting()));
		//
		// double[][] stat = new double[2][countTable.size()];
		// int i = 0;
		// for (long count : countTable.values())
		// stat[0][i++] = count;
		//
		// i = 0;
		// for (String key : countTable.keySet()) {
		// key = key.replace("[", "").replace("]", "");
		// stat[1][i++] = new Profile<>(new Integer[0][],
		// Arrays.stream(key.split(",
		// ")).mapToInt(Integer::parseInt).toArray()).getStat()
		// / Math.pow(numFactorial, numVote) * numSample;
		// }
		//
		// List<String> columnLabel = new ArrayList<>(countTable.size());
		// i = 0;
		// for (; i < countTable.size(); i++)
		// columnLabel.add("P" + (i + 1));
		//
		// Ace ace = new Ace("Preference Profile Sampling");
		// ace.bar(Arrays.asList("Sampling Distribution", "Truth Distribution"),
		// columnLabel, stat);
		//
		TickClock.stopTick();
	}
}