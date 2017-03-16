package com.horsehour.vote.models.noise;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.ExponentialDistribution;

import com.horsehour.vote.Profile;
import com.horsehour.vote.ScoredItems;

/**
 * Gumbel noise model - the model for Plackett-Luce.
 * 
 * @author Andrew Mao
 */
public class GumbelNoiseModel<T> extends RandomUtilityModel<T> {

	public GumbelNoiseModel(ScoredItems<T> scoreMap) {
		super(scoreMap);
	}

	public GumbelNoiseModel(List<T> candidates, double[] strengths) {
		super(candidates, strengths);
	}

	public GumbelNoiseModel(List<T> candidates, double adjStrDiff) {
		super(candidates, adjStrDiff);
	}

	@Override
	public String toParamString() {
		StringBuilder sb = new StringBuilder();

		sb.append(candidates.toString()).append("\n");
		sb.append(Arrays.toString(super.strengthParams));

		return sb.toString();
	}

	@Override
	public double[] sampleUtilities(Random rnd) {
		throw new UnsupportedOperationException("Sampling is currently done with exponentials");
	}

	@Override
	public Profile<T> sampleProfile(int size, Random rnd) {
		T[][] profile = super.getProfileArrayInitialized(size);

		// Set up random number generators
		ExponentialDistribution[] dists = new ExponentialDistribution[candidates.size()];
		for (int j = 0; j < candidates.size(); j++) {
			double mean = Math.exp(-strengthParams[j]);
			dists[j] = new ExponentialDistribution(mean);
			dists[j].reseedRandomGenerator(rnd.nextLong());
		}

		for (int i = 0; i < size; i++) {
			double[] strVals = new double[candidates.size()];

			// Generate exponential random variables
			for (int j = 0; j < candidates.size(); j++)
				strVals[j] = dists[j].sample();

			// Sort by the resulting strength parameters
			sortByStrengthReverse(profile[i], strVals);
		}

		return new Profile<T>(profile);
	}

	@Override
	public double marginalProbability(T winner, T loser) {
		int idxWinner = candidates.indexOf(winner);
		int idxLoser = candidates.indexOf(loser);
		// This is just the logit function
		return 1d / (2 + Math.expm1(strengthParams[idxLoser] - strengthParams[idxWinner]));
	}

	@Override
	public double logLikelihood(Profile<T> profile) {
		// The gumbel model log likelihood.
		double ll = 0;

		for (T[] preference : profile.getData()) {
			double gammaSum = 0;
			for (int i = preference.length - 1; i >= 0; i--) {
				double gamma_i = strengthMap.get(preference[i]).doubleValue();
				gammaSum += Math.exp(gamma_i);
				if (i == preference.length - 1)
					continue;
				ll += gamma_i;
				ll -= Math.log(gammaSum);
			}
		}
		return ll;
	}

}
