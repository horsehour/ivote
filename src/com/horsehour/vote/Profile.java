package com.horsehour.vote;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
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
 * Preference profile over candidates/items by voter. The most important in the
 * structure is all preference ranking of voters. Since many people may share
 * with others the same preference ranking, therefore, we introduce the votes
 * for each preference ranking to reduce the computing time.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Jun. 5, 2016
 */
public class Profile<T> implements Serializable {
	private static final long serialVersionUID = 8603506337214198955L;
	/**
	 * preference rankings the profile contains
	 */
	public T[][] data;
	/**
	 * distribution of votes of corresponding preference rankings, default 1
	 */
	public int[] votes;
	public int numVoteTotal;

	public Profile(T[][] dat) {
		this.data = dat;
		numVoteTotal = dat.length;
		votes = new int[numVoteTotal];
		Arrays.fill(votes, 1);
	}

	public Profile(T[][] dat, int[] votes) {
		this.data = dat;
		this.votes = votes;

		int sum = 0;
		for (int k : votes)
			sum += k;
		numVoteTotal = sum;
	}

	@SuppressWarnings("unchecked")
	public Profile(List<List<T>> dat, int[] votes) {
		int numRanking = dat.size();
		int numItem = dat.get(0).size();

		this.data = (T[][]) Array.newInstance(dat.get(0).get(0).getClass(), new int[] { numRanking, numItem });
		this.votes = votes;
		int sum = 0;
		for (int i = 0; i < numRanking; i++) {
			for (int j = 0; j < numItem; j++)
				data[i][j] = dat.get(i).get(j);
			sum += votes[i];
		}
		this.numVoteTotal = sum;
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

	/**
	 * Rebuild a profile with given items
	 * 
	 * @param items
	 * @return reconstruct preference profile
	 */
	@SuppressWarnings("unchecked")
	public Profile<T> reconstruct(List<T> items) {
		int numRanking = votes.length;
		int numItem = items.size();

		T[][] preferences = (T[][]) Array.newInstance(data[0][0].getClass(), new int[] { numRanking, numItem });
		int[] voteList = new int[numRanking];
		for (int i = 0; i < numRanking; i++) {
			int c = 0;
			for (T item : data[i]) {
				if (items.contains(item))
					preferences[i][c++] = item;
			}
			voteList[i] = votes[i];
		}
		return new Profile<>(preferences, voteList);
	}

	@SuppressWarnings("unchecked")
	public Profile<T> merge(Profile<T> profile) {
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
		T[][] preferences = (T[][]) Array.newInstance(profile.data[0][0].getClass(), new int[] { numRanking, numItem });
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
		return new Profile<T>(preferences, voteList);
	}

	@SuppressWarnings("unchecked")
	public Profile<T> compact() {
		int[] fingerprint = getFingerPrint();

		// distinct hash codes
		List<Integer> codes = new ArrayList<>();
		for (int code : fingerprint)
			codes.add(code);
		codes = codes.stream().distinct().sorted().collect(Collectors.toList());

		int m = getNumItem();
		int d = codes.size();

		T[][] preferences = (T[][]) Array.newInstance(data[0][0].getClass(), new int[] { d, m });
		int[] nv = new int[d];
		for (int i = 0; i < d; i++) {
			int k = 0;
			for (int code : fingerprint) {
				if (code == codes.get(i)) {
					if (nv[i] == 0)
						preferences[i] = Arrays.copyOf(data[k], m);
					nv[i]++;
				}
				k++;
			}
		}
		return new Profile<T>(preferences, nv);
	}

	public T[][] getData() {
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
			int remain = numVoteTotal - sum;
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
			int remain = numVoteTotal - sum;
			sum += vote;
			prod *= CombinatoricsUtils.binomialCoefficient(remain, vote) / Math.pow(factorial, vote);
		}
		return prod;
	}

	public List<T> getSortedOrdering() {
		return Arrays.asList(getSortedItems());
	}

	public T[] getSortedItems() {
		T[] candidates = Arrays.copyOf(data[0], data[0].length);
		Arrays.sort(candidates);
		return candidates;
	}

	public int getNumItem() {
		return data[0].length;
	}

	public int getNumVote() {
		return numVoteTotal;
	}

	/**
	 * Get the indices of preferences with respect to some ordering
	 * 
	 * @param ordering
	 * @return
	 */
	public List<int[]> getIndices(List<T> ordering) {
		List<int[]> rankings = new ArrayList<>();

		for (T[] preference : data) {
			int[] ranking = new int[preference.length];
			int i = 0;
			for (T item : preference)
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
	public Map<T, int[]> getPositionCounts() {
		Map<T, int[]> counts = new TreeMap<>();

		for (T t : data[0])
			counts.put(t, new int[data[0].length]);

		for (T[] ranking : data) {
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
	public int getNumCorrect(T first, T second, Comparator<T> comp) {
		int correct = 0;

		for (T[] ranking : data) {
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
	public Profile<T> copyRandomSubset(int subsetSize, Random rnd) {
		if (subsetSize < data.length)
			return new Profile<T>(Sampling.selectKRandom(data, subsetSize, rnd));
		else
			return this;
	}

	public Profile<T> slice(int fromIdx, int toIdx) {
		int newSize = toIdx - fromIdx;
		@SuppressWarnings("unchecked")
		T[][] newProfile = (T[][]) Array.newInstance(data[0][0].getClass(), data.length, newSize);

		for (int i = 0; i < data.length; i++)
			System.arraycopy(data[i], fromIdx, newProfile[i], 0, newSize);

		return new Profile<T>(newProfile);
	}

	@SuppressWarnings("unchecked")
	public static <T> Profile<T> singleton(List<T> ranking) {
		T[] arr = ranking.toArray((T[]) Array.newInstance(ranking.get(0).getClass(), ranking.size()));
		T[][] arr2 = (T[][]) Array.newInstance(arr.getClass(), 1);
		arr2[0] = arr;
		return new Profile<>(arr2);
	}

	public boolean equals(Profile<T> profile) {
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