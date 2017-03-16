package com.horsehour.vote.models;

import java.util.List;

import com.horsehour.vote.models.noise.NoiseModel;
import com.horsehour.vote.models.noise.OrdinalEstimator;

public abstract class RandomUtilityEstimator<M extends NoiseModel<?>, P> implements OrdinalEstimator<M> {

	public abstract P getParameters(List<int[]> rankings, int numItems);

	public String getName() {
		return getClass().getSimpleName();
	}
}
