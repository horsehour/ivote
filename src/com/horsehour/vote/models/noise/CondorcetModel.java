package com.horsehour.vote.models.noise;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

import com.horsehour.vote.Profile;
import com.horsehour.vote.Sampling;
import com.horsehour.vote.metric.RankingMetric;

/**
 * Generates preference profiles with Condorcet noise, aka Mallows model
 * 
 * @author mao
 */
public class CondorcetModel<T> extends NoiseModel<T> {

	final double phi;
	final Collection<List<T>> candidatesMixed;

	public CondorcetModel(Collection<List<T>> candidatesMixed, double prob) {
		super(candidatesMixed.iterator().next());
		this.candidatesMixed = candidatesMixed;

		if (prob > 1.0 || prob < 0.5)
			throw new IllegalArgumentException("p must be in the range [0.5, 1]");

		this.phi = (1.0 - prob) / prob;
	}

	public CondorcetModel(List<T> candidates, double prob) {
		super(candidates);
		candidatesMixed = null;

		if (prob > 1.0 || prob < 0.5)
			throw new IllegalArgumentException("p must be in the range [0.5, 1]");

		this.phi = (1.0 - prob) / prob;
	}

	public Collection<List<T>> getCandidateMixture() {
		return candidatesMixed;
	}

	public String toString() {
		return candidates.toString() + " p=" + 1 / (1 + phi);
	}

	@Override
	public String toParamString() {
		StringBuilder sb = new StringBuilder();

		if (candidatesMixed != null) {
			for (List<T> candidates : candidatesMixed) {
				sb.append(candidates.toString()).append("\n");
			}
		} else
			sb.append(candidates.toString());

		sb.append(1 / (1 + phi));
		return sb.toString();
	}

	static double[] getWeights(int i, double phi) {
		// Draw integers from 0 .. i with probability p_ij as defined in Lu &
		// Boutilier paper
		double[] wts = new double[i + 1];
		for (int j = 0; j <= i; j++)
			wts[j] = Math.pow(phi, i - j);
		return wts;
	}

	int[] getNewInsertionVector(Random rnd) {
		int[] insVec = new int[candidates.size()];

		for (int i = 0; i < insVec.length; i++) {
			double[] wts = getWeights(i, phi);

			insVec[i] = Sampling.selectRandomWeighted(wts, rnd);
		}

		return insVec;
	}

	List<T> getInsertedList(int[] insVec) {
		// Insert candidates at the position specified by the insertion vector
		List<T> ranking = new ArrayList<>();
		for (int j = 0; j < candidates.size(); j++)
			ranking.add(insVec[j], candidates.get(j));

		return ranking;
	}

	@Override
	public Profile<T> sampleProfile(int size, Random rnd) {
		T[][] profile = super.getProfileArray(size);
		for (int i = 0; i < size; i++) {
			int[] insVec = getNewInsertionVector(rnd);
			List<T> ranking = getInsertedList(insVec);
			ranking.toArray(profile[i]);
		}
		return new Profile<T>(profile);
	}

	@Override
	public double computeMLMetric(RankingMetric<T> metric) {
		if (candidatesMixed != null)
			return metric.computeAverage(candidatesMixed);
		else
			return metric.compute(candidates);
	}

	@Override
	public double marginalProbability(T winner, T loser) {
		if (candidatesMixed != null) {
			Mean m = new Mean();
			for (List<T> ranking : candidatesMixed)
				m.increment(mallowsPairwiseProb(ranking.indexOf(loser) - ranking.indexOf(winner), phi));
			return m.getResult();
		} else
			return mallowsPairwiseProb(candidates.indexOf(loser) - candidates.indexOf(winner), phi);
	}

	static double mallowsPairwiseProb(int difference, double phi) {
		int c = difference > 0 ? difference : -difference;
		double num = 1, denom1 = 1, denom2 = 1;

		double phi_k = 1;
		for (int k = 1; k < c; k++) {
			phi_k *= phi;
			num += (k + 1) * phi_k;
			denom1 += phi_k;
			denom2 += phi_k;
		}
		// Extra iteration for second part of denom
		denom2 += phi_k * phi;

		double prob = num / denom1 / denom2;
		return difference > 0 ? prob : 1 - prob;
	}

	@Override
	public double logLikelihood(Profile<T> profile) {
		if (candidatesMixed == null)
			return profileLogLikelihood(profile, candidates, phi);

		// Need to do log-sum-exp here
		double[] logProbs = new double[candidatesMixed.size()];
		int i = 0;
		for (List<T> candidates : candidatesMixed)
			logProbs[i++] = profileLogLikelihood(profile, candidates, phi);
		// Average the sum by subtracting log(n)
		return getLogSumExp(logProbs) - Math.log(logProbs.length);
	}

	/**
	 * Computes log(sum{i in input} exp(a_i)).
	 * 
	 * We shift by max instead of average since log probabilities are always
	 * negative. expm1 is used since log probs are shifted closer to 0.
	 * 
	 * @see http://jblevins.org/notes/log-sum-exp
	 * @param logProbs
	 * @return
	 */
	private double getLogSumExp(double[] logProbs) {
		double max = Double.NEGATIVE_INFINITY;
		for (double d : logProbs)
			max = Math.max(d, max);

		double sum = 0;
		for (double d : logProbs)
			sum += Math.expm1(d - max) + 1;

		return max + Math.log(sum);
	}

	public static <T> double profileLogLikelihood(Profile<T> profile, List<T> trueRanking, double phi) {
		double ll = 0;

		for (T[] preference : profile.getData()) {
			/*
			 * Compute probability as would occur by insertion First candidate
			 * has insertion probability 1
			 */

			for (int i = 1; i < trueRanking.size(); i++) {
				T alternative = trueRanking.get(i);

				int insertionIdx = ArrayUtils.indexOf(preference, alternative);
				int diff = 0;
				// Subtract 1 for everything that comes after i in the true
				// ranking
				for (int j = 0; j < insertionIdx; j++) {
					if (trueRanking.indexOf(preference[j]) > i)
						diff++;
				}

				ll += Math.log(getInsertionProbs(i, phi)[insertionIdx - diff]);
			}
		}

		return ll;
	}

	private static double[] getInsertionProbs(int i, double phi) {
		double[] weights = getWeights(i, phi);

		double sum = 0;
		for (double d : weights)
			sum += d;
		for (int j = 0; j < weights.length; j++)
			weights[j] /= sum;

		return weights;
	}

}
