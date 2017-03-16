package com.horsehour.vote.models;

import java.util.List;

import com.horsehour.vote.Profile;
import com.horsehour.vote.models.noise.NoiseModel;
import com.horsehour.vote.models.noise.OrdinalEstimator;

public abstract class PairwiseDiscreteChoiceEstimator<M extends NoiseModel<?>> implements OrdinalEstimator<M> {

	protected <T> double[][] addAdjacentPairs(Profile<T> profile, List<T> ordering) {
		int m = ordering.size();
		double[][] wins = new double[m][m];

		for (T[] ranking : profile.getData()) {
			for (int i = 0; i < ranking.length - 1; i++) {
				int idxWinner = ordering.indexOf(ranking[i]);
				int idxLoser = ordering.indexOf(ranking[i + 1]);

				wins[idxWinner][idxLoser] += 1;
			}
		}

		return wins;
	}

	protected <T> double[][] addAllPairs(Profile<T> profile, List<T> ordering) {
		int m = ordering.size();
		double[][] wins = new double[m][m];

		for (T[] ranking : profile.getData()) {
			for (int i = 0; i < ranking.length; i++) {
				for (int j = i + 1; j < ranking.length; j++) {
					int idxWinner = ordering.indexOf(ranking[i]);
					int idxLoser = ordering.indexOf(ranking[j]);

					wins[idxWinner][idxLoser] += 1;
				}
			}
		}

		return wins;
	}

	@Override
	public <T> M fitModelOrdinal(Profile<T> profile) {
		return this.fitModel(profile, true);
	}

	public abstract <T> M fitModel(Profile<T> profile, boolean useAllPairs);

	public abstract double[] getParameters(double[][] winMatrix);
}
