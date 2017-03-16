package com.horsehour.vote.models.noise;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;
import com.horsehour.vote.metric.RankingMetric;

public abstract class RandomUtilityModel<T> extends NoiseModel<T> {

	protected final ScoredItems<T> strengthMap;
	protected final double[] strengthParams;

	public RandomUtilityModel(ScoredItems<T> strengths) {
		super(new ArrayList<T>(strengths.keySet()));
		this.strengthMap = strengths;

		strengthParams = new double[candidates.size()];

		for (int j = 0; j < strengthParams.length; j++)
			strengthParams[j] = strengthMap.get(candidates.get(j)).doubleValue();
	}

	public RandomUtilityModel(List<T> candidates, double[] strengthParams) {
		super(candidates);

		if (candidates.size() != strengthParams.length)
			throw new RuntimeException("Must have same number of strength parameters as candidates");

		this.strengthMap = new ScoredItems<T>(candidates, strengthParams);
		this.strengthParams = strengthParams;
	}

	public RandomUtilityModel(List<T> candidates, double adjStrengthDiff) {
		super(candidates);

		this.strengthMap = new ScoredItems<T>(candidates);
		this.strengthParams = new double[candidates.size()];

		for (int i = 0; i < strengthParams.length; i++) {
			double strength = -i * adjStrengthDiff;
			strengthParams[i] = strength;
			strengthMap.put(candidates.get(i), strength);
		}
	}

	public double[] getValues() {
		return strengthParams;
	}

	public ScoredItems<T> getValueMap() {
		return strengthMap;
	}

	/**
	 * Sample random utilities in the same order as the initialized candidates
	 * 
	 * @param rnd
	 * @return
	 */
	public abstract double[] sampleUtilities(Random rnd);

	@Override
	public Profile<T> sampleProfile(int size, Random rnd) {
		T[][] profile = super.getProfileArrayInitialized(size);

		for (int i = 0; i < size; i++) {
			double[] strVals = sampleUtilities(rnd);

			// Sort by the resulting strength parameters
			sortByStrengths(profile[i], strVals);
		}

		return new Profile<T>(profile);
	}

	// Higher strength parameter comes earlier in the array
	void sortByStrengths(T[] arr, final double[] strengths) {
		Comparator<T> comparator = (o1, o2) -> {
			int i1 = candidates.indexOf(o1);
			int i2 = candidates.indexOf(o2);
			return Double.compare(strengths[i2], strengths[i1]);
		};
		Arrays.sort(arr, comparator);
	}

	/*
	 * Reverse sort order - lower exponential comes first so it's the same as
	 * normal sort order
	 */
	void sortByStrengthReverse(T[] arr, final double[] strengths) {
		Comparator<T> comparator = (o1, o2) -> {
			int i1 = candidates.indexOf(o1);
			int i2 = candidates.indexOf(o2);
			return Double.compare(strengths[i1], strengths[i2]);
		};
		Arrays.sort(arr, comparator);
	}

	public double computeMLMetric(RankingMetric<T> metric) {
		return metric.computeByScore(strengthMap);
	}
}
