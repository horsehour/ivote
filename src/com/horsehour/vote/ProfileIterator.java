package com.horsehour.vote;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.collections4.iterators.PermutationIterator;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 6:51:14 PM, Jun 25, 2016
 *
 */
public class ProfileIterator<T> implements Iterator<Profile<T>> {
	private List<List<T>> permutationList;
	private Selection<Integer> indexIter;

	public ProfileIterator(List<T> itemList, int numVote, boolean named) {
		PermutationIterator<T> iter = new PermutationIterator<>(itemList);
		permutationList = new ArrayList<>();
		List<Integer> pidList = new ArrayList<>();
		int count = 0;
		while (iter.hasNext()) {
			permutationList.add(iter.next());
			pidList.add(count);
			count++;
		}
		indexIter = new Selection<>(pidList, numVote, named);
	}

	public ProfileIterator(T[] items, int numVote, boolean named) {
		PermutationIterator<T> iter = new PermutationIterator<>(Arrays.asList(items));
		permutationList = new ArrayList<>();
		List<Integer> pidList = new ArrayList<>();
		int count = 0;
		while (iter.hasNext()) {
			permutationList.add(iter.next());
			pidList.add(count);
			count++;
		}
		indexIter = new Selection<>(pidList, numVote, named);
	}

	public boolean hasNext() {
		return indexIter.hasNext();
	}

	public Profile<T> next() {
		List<Integer> preference = indexIter.next();
		Set<Entry<Integer, Long>> entries = preference.stream()
				.collect(Collectors.groupingBy(e -> e, Collectors.counting())).entrySet();

		int numRanking = entries.size();
		List<List<T>> data = new ArrayList<>();

		Iterator<Entry<Integer, Long>> iter = entries.iterator();
		int index = 0;

		int[] votes = new int[numRanking];
		while (iter.hasNext()) {
			Entry<Integer, Long> entry = iter.next();
			int pid = entry.getKey();
			data.add(permutationList.get(pid));
			votes[index] = entry.getValue().intValue();
			index++;
		}
		return new Profile<>(data, votes);
	}

	public static void main(String[] args) {
		List<List<Integer>> permutationList;
		Selection<Integer> indexIter;

		int numVote = 5;
		PermutationIterator<Integer> iter = new PermutationIterator<>(Arrays.asList(0, 1, 2));

		permutationList = new ArrayList<>();
		List<Integer> pidList = new ArrayList<>();
		int count = 0;
		while (iter.hasNext()) {
			permutationList.add(iter.next());
			pidList.add(count);
			count++;
		}
		indexIter = new Selection<>(pidList, numVote, false);

		//permutation id, votes
		Map<Integer, Long> preferences = null;
		while (indexIter.hasNext())
			preferences = indexIter.next().stream().collect(Collectors.groupingBy(e -> e, Collectors.counting()));
		
		System.out.println(preferences.size());
	}
}
