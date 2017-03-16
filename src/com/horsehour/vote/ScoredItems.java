package com.horsehour.vote;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Jun 8, 2016
 * 
 */
public class ScoredItems<T> extends LinkedHashMap<T, Double> {

	private static final long serialVersionUID = -6171230363427632120L;

	public ScoredItems() {
	}

	public ScoredItems(List<T> items, double[] scores) {
		super();

		if (items.size() != scores.length)
			throw new IllegalArgumentException("Items must have a corresponding array of values");

		for (int i = 0; i < scores.length; i++)
			put(items.get(i), scores[i]);
	}

	public ScoredItems(T[] items, double[] scores) {
		super();

		if (items.length != scores.length)
			throw new IllegalArgumentException("Items must have a corresponding array of values");

		for (int i = 0; i < scores.length; i++)
			put(items[i], scores[i]);
	}

	public ScoredItems(T[] items, int[] scores) {
		super();

		if (items.length != scores.length)
			throw new IllegalArgumentException("Items must have a corresponding array of values");

		for (int i = 0; i < scores.length; i++)
			put(items[i], 1.0d * scores[i]);
	}

	/**
	 * Creates items with a default score of 0
	 * 
	 * @param items
	 */
	public ScoredItems(Collection<T> items) {
		super();

		for (T item : items)
			put(item, 0.0);
	}

	/**
	 * Creates items with a default score of 0
	 * 
	 * @param items
	 */
	public ScoredItems(T[] items) {
		super();

		for (int i = 0; i < items.length; i++)
			put(items[i], 0.0);
	}

	/**
	 * @return ranking of items in order
	 */
	public List<T> getRanking() {
		List<T> ranking = new ArrayList<>(keySet());
		// Sort items in descending order using their scores. Top Ties May Exist!
		ranking.sort((t1, t2) -> get(t2).compareTo(get(t1)));
		return ranking;
	}

	/**
	 * @return an array in the same order as the original items.
	 */
	public double[] toArray() {
		double[] scores = new double[size()];
		int i = 0;
		for (double score : values())
			scores[i++] = score;
		return scores;
	}
}