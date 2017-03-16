package com.horsehour.vote.models.noise;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.collections4.iterators.PermutationIterator;

import com.horsehour.vote.Profile;
import com.horsehour.vote.metric.RankingMetric;

/**
 * The random generation of a profile of voter preferences is usually called a
 * culture. For instance choosing n preferences uniformly and independently
 * among the m! linear orderings of m alternatives is called the impartial
 * culture of size (n, m).
 * 
 * @see Handbook on Approval Voting, Jean-François Laslier and M. Remzi Sanver
 * 
 * @author Andrew Mao
 * @author Chunheng Jiang
 * @version 1.0
 * @since Jun 6, 2016
 */
public abstract class NoiseModel<T> {

	protected List<T> candidates;
	protected Double fittedLikelihood = null;

	public NoiseModel(List<T> candidates) {
		this.candidates = candidates;
	}

	@SuppressWarnings("unchecked")
	protected T[][] getProfileArray(int size) {
		int[] dimensions = new int[] { size, candidates.size() };
		return (T[][]) Array.newInstance(candidates.get(0).getClass(), dimensions);
	}

	protected T[][] getProfileArrayInitialized(int size) {
		T[][] profile = getProfileArray(size);

		for (T[] ranking : profile)
			candidates.toArray(ranking);
		return profile;
	}

	/**
	 * Sample a preference profile from this model
	 * 
	 * @param size
	 * @param rnd
	 * @return
	 */
	public abstract Profile<T> sampleProfile(int size, Random rnd);

	/**
	 * Compute the goodness of this model by some ranking metric
	 * 
	 * @param metric
	 * @return
	 */
	public abstract double computeMLMetric(RankingMetric<T> metric);

	/**
	 * Compute the goodness of this model over all implied probabilistic
	 * rankings
	 * 
	 * @param metric
	 * @return
	 */
	public double computeExpectedMetric(RankingMetric<T> metric) {
		int m = candidates.size();
		PermutationIterator<Integer> iter = null;
		List<Integer> list = IntStream.range(0, m).boxed().collect(Collectors.toList());
		iter = new PermutationIterator<Integer>(list);

		// Average the metric weighted by the probability of each permutation
		double value = 0;
		while (iter.hasNext()) {
			List<T> permutation = new ArrayList<T>(m);
			for (int i : iter.next())
				permutation.add(candidates.get(i));

			Profile<T> singleton = Profile.singleton(permutation);
			value += Math.exp(logLikelihood(singleton)) * metric.compute(permutation);
		}

		return value;
	}

	/**
	 * The probability of one candidate beating another under this noise model.
	 * 
	 * @param winner
	 * @param loser
	 * @return
	 */
	public abstract double marginalProbability(T winner, T loser);

	public void setFittedLikelihood(double ll) {
		fittedLikelihood = ll;
	}

	public Double getFittedLikelihood() {
		return fittedLikelihood;
	}

	public abstract double logLikelihood(Profile<T> profile);

	/**
	 * Serialize the model to a string.
	 * 
	 * @return
	 */
	public abstract String toParamString();

}
